from __future__ import annotations

# fmt: off
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from ..io_utils import (
    read_json_cache as io_read_json_cache,
    write_json_cache as io_write_json_cache,
    _resolve_safe_path,
)

# Best-effort import of config helpers that provide deep_research profile/budget
# normalization for cache-key determinism. If unavailable at runtime we fall back
# to legacy behavior (no resolved profile injection).
try:
    from ..config import CONFIG, deep_research_profile_for_cache  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    CONFIG = None  # type: ignore
    deep_research_profile_for_cache = None  # type: ignore

# NOTE: The canonical event emitter API lives in aidev/events.py and should export
# emit_event(event_type: str, payload: Any, session_id: Optional[str]=None).
# We attempt a best-effort import but also accept a few alternative public names
# to be robust across minor versions while preferring the canonical alias.
try:
    # Preferred canonical alias (decisions require this name to be exported)
    from ..events import emit_event as _emit_event  # type: ignore
except Exception:  # pragma: no cover - runtime fallback
    try:
        # Common alternative name
        from ..events import send_event as _emit_event  # type: ignore
    except Exception:
        try:
            # Import module and pick first available emitter-like attribute
            from .. import events as _events  # type: ignore

            if hasattr(_events, "emit_event"):
                _emit_event = getattr(_events, "emit_event")
            elif hasattr(_events, "send_event"):
                _emit_event = getattr(_events, "send_event")
            elif hasattr(_events, "emit"):
                _emit_event = getattr(_events, "emit")
            elif hasattr(_events, "_emit"):
                # internal helper available; accept as last resort
                _emit_event = getattr(_events, "_emit")
            else:
                _emit_event = None  # type: ignore
        except Exception:
            _emit_event = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "write_json_cache",
    "read_json_cache",
    "merge_meta_budgets",
    "impact_surface_for_paths",
    "compute_cache_key",
    "get_cache_metadata",
    "normalize_budget_for_cache",
]


_VOLATILE_META_KEYS = {
    "ts",
    "generated_at",
    "created_at",
    "updated_at",
}


def _extract_artifact_digest(rel_cache: Path) -> Optional[str]:
    """Extract the deterministic 12-char digest from filenames like name--<digest>.json.

    Returns None if the pattern isn't present.
    """
    try:
        name = Path(rel_cache).name
        stem = Path(name).stem  # if name is foo--abc123.json -> stem is foo--abc123
        if "--" not in stem:
            return None
        digest = stem.rsplit("--", 1)[-1]
        if len(digest) == 12 and all(c in "0123456789abcdef" for c in digest.lower()):
            return digest
    except Exception:
        return None
    return None


def _sanitize_meta_for_event(meta: Any) -> dict:
    """Return small, deterministic meta payload for events (no volatile keys)."""
    if not isinstance(meta, dict):
        return {}
    out: dict = {}
    # Only include shallow budget (sanitized) for cache events.
    b = meta.get("budget")
    if isinstance(b, dict):
        b2 = dict(b)
        for k in list(b2.keys()):
            if k in _VOLATILE_META_KEYS:
                b2.pop(k, None)
        out["budget"] = b2
    # Include truncation summary (if present) but avoid raw content.
    t = meta.get("truncated")
    if isinstance(t, dict):
        out["truncated"] = dict(t)
    # Include errors (best-effort) for observability.
    errs = meta.get("errors")
    if isinstance(errs, list):
        out["errors"] = [str(e) for e in errs[:10]]
    return out


def _emit_deep_research_event(event_type: str, payload: dict) -> None:
    """Best-effort structured event emission. Never raises."""
    if _emit_event is None:
        # Emit a conspicuous debug/warning so missing emitter is visible in smoke runs.
        logger.warning(
            "Event emitter 'emit_event' not available in aidev.events; skipping event '%s' with payload keys=%s",
            event_type,
            list(payload.keys()),
        )
        return
    try:
        # Ensure we have a callable before calling
        if callable(_emit_event):
            _emit_event(event_type, payload)
            logger.debug("Emitted event %s", event_type)
        else:
            logger.warning("Found non-callable event emitter; skipping %s", event_type)
    except Exception:
        logger.exception("Failed to emit event %s", event_type)


def _strip_volatile_meta_for_cache(request: dict) -> dict:
    """Return a shallow-copied request with volatile meta fields removed.

    This function is intentionally conservative: it only strips well-known timestamp-ish
    keys under request['meta'].

    Does not mutate the input.
    """
    if not isinstance(request, dict):
        return request

    req = dict(request)
    meta = req.get("meta")
    if not isinstance(meta, dict):
        return req

    new_meta = dict(meta)
    removed: list[str] = []
    for k in list(new_meta.keys()):
        if k in _VOLATILE_META_KEYS:
            removed.append(k)
            new_meta.pop(k, None)

    if removed:
        logger.debug("Stripped volatile meta keys for cache key: %s", removed)

    req["meta"] = new_meta
    return req


def normalize_budget_for_cache(budget: dict) -> dict:
    """Normalize a budget dict for deterministic cache key computation.

    Goals:
    - Remove only well-known volatile keys (e.g., 'ts', 'created_at') if present.
    - Coerce numeric types consistently (e.g., 1.0 -> 1) to avoid float formatting drift.
    - Keep structure otherwise intact; do not reorder semantically-ordered lists.

    Does not mutate the input.
    """
    if not isinstance(budget, dict):
        return {}

    b = dict(budget)

    removed: list[str] = []
    for k in list(b.keys()):
        if k in _VOLATILE_META_KEYS:
            removed.append(k)
            b.pop(k, None)

    if removed:
        logger.debug("Removed volatile budget keys for cache key: %s", removed)

    # Coerce numeric types deterministically.
    # - If float is integral, convert to int.
    # - Recurse into nested dicts.
    for k, v in list(b.items()):
        if isinstance(v, float) and v.is_integer():
            b[k] = int(v)
        elif isinstance(v, dict):
            b[k] = normalize_budget_for_cache(v)
        # Do not normalize list order: may be semantically ordered.

    # Deterministic key order for JSON serialization (json.dumps(sort_keys=True) will sort,
    # but returning a sorted dict makes debugging/logging stable).
    return dict(sorted(b.items(), key=lambda kv: kv[0]))


def compute_cache_key(
    request: dict,
    repo_brief_hash: str,
    *,
    resolved_profile: Optional[str] = None,
    resolved_budget: Optional[dict] = None,
) -> str:
    """Compute a short deterministic cache key for a request.

    Determinism contract:
    - The request is canonicalized via JSON serialization with sorted keys and compact separators.
    - Unicode is preserved (ensure_ascii=False) and the bytes are UTF-8 encoded.

    This helper also strips volatile meta timestamp fields (e.g., meta.ts) so two logically
    identical requests that differ only by timestamps map to the same key.

    Call compatibility:
    - Existing callers may continue to call compute_cache_key(request, repo_brief_hash).
    - New callers may pass resolved_profile/resolved_budget (keyword-only) to ensure the
      resolved profile/budget are incorporated deterministically.

    Note: cache artifacts written by this module may include a meta.ts timestamp field
    (ISO8601 UTC, no subseconds). Determinism tests comparing artifact content should
    ignore meta.ts.
    """
    import json

    # Copy and strip volatile meta before computing any derived fields
    req_no_volatile = _strip_volatile_meta_for_cache(request)

    profile: str = ""
    budget_norm: dict = {}

    if resolved_profile is not None:
        profile = str(resolved_profile)
    else:
        try:
            meta = req_no_volatile.get("meta") if isinstance(req_no_volatile, dict) else None
            if isinstance(meta, dict) and meta.get("profile") is not None:
                profile = str(meta.get("profile"))
        except Exception:
            # Best-effort only
            profile = ""

    if resolved_budget is not None and isinstance(resolved_budget, dict):
        budget_norm = normalize_budget_for_cache(resolved_budget)
        logger.debug("Normalized resolved_budget for cache key (keys=%s)", list(budget_norm.keys()))
    else:
        # Best-effort extraction from request.meta.budget
        try:
            meta = req_no_volatile.get("meta") if isinstance(req_no_volatile, dict) else None
            if isinstance(meta, dict):
                b = meta.get("budget")
                if isinstance(b, dict):
                    budget_norm = normalize_budget_for_cache(b)
                    logger.debug(
                        "Normalized request meta.budget for cache key (keys=%s)",
                        list(budget_norm.keys()),
                    )
        except Exception:
            budget_norm = {}

    # Attempt to resolve a canonical deep_research profile object for cache-key stability.
    # If aidev.config.deep_research_profile_for_cache is available, include its output
    # (normalized) so changes in configured profiles/caps affect cache keys deterministically.
    resolved_profile_for_cache = None
    try:
        if deep_research_profile_for_cache is not None and CONFIG is not None:
            try:
                rp = deep_research_profile_for_cache(cfg=CONFIG, profile_name=profile)
                if isinstance(rp, dict):
                    # Normalize structure for deterministic serialization
                    resolved_profile_for_cache = normalize_budget_for_cache(rp)
                elif rp is not None:
                    resolved_profile_for_cache = str(rp)
            except Exception:
                logger.exception("deep_research_profile_for_cache failed; proceeding without resolved profile")
    except Exception:
        # Any unexpected failure should not break key computation
        resolved_profile_for_cache = None

    # Canonical wrapper ensures budget/profile impact the key deterministically.
    wrapper = {
        "request": req_no_volatile,
        "profile": profile,
        "budget": budget_norm,
        "resolved_profile_for_cache": resolved_profile_for_cache,
    }

    canonical = json.dumps(wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    key_source = f"{canonical}||{repo_brief_hash}"
    return hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:12]


def _merge_budgets(existing: dict, incoming: dict, merge_keys: Optional[list] = None) -> dict:
    """
    Deterministically merge two budget dicts.

    - If merge_keys is provided, for those keys prefer the numeric maximum when both
      values are numeric. For non-numeric or missing values, prefer the incoming value.
    - If merge_keys is not provided, any numeric field present in both will take the max.
    - For other keys, incoming takes precedence.

    This function does not mutate inputs; it returns a new merged dict.
    """
    merged = dict(existing) if existing is not None else {}
    incoming = dict(incoming) if incoming is not None else {}

    # If merge_keys provided, use that set; otherwise infer numeric keys present in both
    if merge_keys:
        keys_to_compare = set(merge_keys)
    else:
        # Only consider keys that are present in both and are numeric in both inputs
        keys_to_compare = set(
            k
            for k in (merged.keys() & incoming.keys())
            if isinstance(merged.get(k), (int, float)) and isinstance(incoming.get(k), (int, float))
        )

    for k, v in incoming.items():
        if k in keys_to_compare:
            old_v = merged.get(k)
            # If both numeric, choose max deterministically
            if isinstance(old_v, (int, float)) and isinstance(v, (int, float)):
                merged[k] = max(old_v, v)
            else:
                # Prefer incoming for non-numeric or when not both numeric
                merged[k] = v
        else:
            # Default: incoming wins for non-specified keys
            merged[k] = v

    return merged


def _clamp_budget_values(budget: dict, plan_caps: Optional[dict] = None) -> dict:
    """
    Clamp numeric budget values to be >= 0 and, if plan_caps provided, not exceed the caps.
    Does not mutate the input; returns a new dict.
    """
    res = dict(budget) if budget is not None else {}
    caps = dict(plan_caps) if plan_caps is not None else None

    for k, v in list(res.items()):
        if isinstance(v, (int, float)):
            val = v
            if caps is not None:
                cap_val = caps.get(k)
                if isinstance(cap_val, (int, float)):
                    val = min(val, cap_val)
            # Enforce non-negative
            res[k] = max(0, val)
    return res


def merge_meta_budgets(old: dict, new: dict, merge_strategy: str | None = None, merge_keys: list | None = None) -> dict:
    """
    Public helper to merge budget metadata.

    - merge_strategy: 'deterministic' will deterministically merge numeric fields (max on merge_keys or any numeric overlap).
      'override' or any other falsy value will cause the new budget to take precedence (new wins).
    - merge_keys: optional list of keys to apply deterministic max behavior to (when strategy is 'deterministic').

    After merging, numeric budget fields are clamped to be >= 0 and, if plan caps are provided
    (in the new budget or falling back to the old budget), to not exceed those caps.

    The returned dict is a new object and inputs are not mutated.
    """
    old = dict(old) if old is not None else {}
    new = dict(new) if new is not None else {}

    # Determine effective strategy: explicit param takes precedence, otherwise new's merge_strategy
    effective_strategy = merge_strategy or new.get("merge_strategy")

    # Determine plan caps: prefer new's plan_caps if present, otherwise fallback to old's
    plan_caps = None
    if isinstance(new.get("plan_caps"), dict):
        plan_caps = new.get("plan_caps")
    elif isinstance(old.get("plan_caps"), dict):
        plan_caps = old.get("plan_caps")

    if effective_strategy == "deterministic":
        merged = _merge_budgets(old, new, merge_keys)
        merged = _clamp_budget_values(merged, plan_caps)
        return merged

    # 'override' or default: new wins; still apply clamping
    merged = dict(new)
    merged = _clamp_budget_values(merged, plan_caps)
    return merged


def _deterministic_rel_cache(
    project_root: str | Path,
    rel_path: str | Path,
    rec_id: Optional[str] = None,
    *,
    cache_key: Optional[str] = None,
) -> Path:
    """Compute a deterministic repository-local cache path under .aidev/cache.

    If cache_key is provided, it is preferred as the stable suffix seed; otherwise, rec_id
    (if provided) is used. Callers that need stable, reproducible artifact filenames
    (for example Phase 0 preflight artifacts) should pass an explicit cache_key.

    The digest is derived from the canonical project_root, the chosen key, and the
    rel_path to ensure stable, per-run keys that do not collide across repositories.

    Examples:
      rel_path = Path('evidence/basic') -> .aidev/cache/evidence/basic.json (if no suffix exists, .json will not be forced)
      rel_path = Path('evidence/basic.json'), cache_key = 'abc123' -> .aidev/cache/evidence/basic--<digest>.json

    The function never returns paths outside the .aidev/cache tree; callers must still
    validate with _resolve_safe_path(project_root, rel_cache).
    """
    project_root = Path(project_root)
    rel = Path(rel_path)

    # Ensure rel is treated as a repository-relative path; if an absolute path was provided
    # strip its anchor/root so we don't accidentally produce a path that ignores the
    # .aidev/cache prefix when joined using Path.
    if rel.is_absolute():
        try:
            # Relativize against the anchor (works for Unix and Windows drives)
            rel = rel.relative_to(rel.anchor)
        except Exception:
            # Fallback to stripping the first part if relative_to() fails for some reason
            rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)

    cache_parent = Path(".aidev") / "cache" / rel.parent
    name = rel.name

    # Prefer explicit cache_key over rec_id so callers can deterministically opt-in
    # to stable artifact filenames by providing cache_key.
    seed_key = cache_key if cache_key is not None else rec_id
    if seed_key:
        # Create a short deterministic digest from the canonical project root and key
        try:
            canonical = str(project_root.resolve())
        except Exception:
            # Fallback to the provided string form if resolve() fails
            canonical = str(project_root)
        key_source = f"{canonical}||{seed_key}||{str(rel)}"
        digest = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:12]
        stem = Path(name).stem
        suffix = Path(name).suffix or ""
        # Ensure suffix is preserved (if none provided, don't force .json here)
        name = f"{stem}--{digest}{suffix}"

    return cache_parent / name


def _first_present(mapping: dict, keys: list[str]) -> Any:
    """Return the first non-None value among mapping[key] for the given keys."""
    for k in keys:
        try:
            v = mapping.get(k)
            if v is not None:
                return v
        except Exception:
            continue
    return None


def _coerce_int(v: Any) -> Optional[int]:
    """Best-effort coercion to a non-negative int. Returns None on failure."""
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return max(0, v)
        if isinstance(v, float) and v.is_integer():
            return max(0, int(v))
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            return max(0, int(float(s)))
    except Exception:
        return None
    return None


def _apply_artifact_caps(obj: dict, budget: dict) -> tuple[dict, bool, Optional[dict], Optional[str]]:
    """Apply deterministic, conservative caps to a deep research artifact.

    Caps are derived from meta.budget. Support common alias keys so this file can
    remain compatible across versions/config surfaces.

    Expected/aliased keys (to be aligned with aidev/config.py):
      - max_evidence / max_evidence_items / max_evidence_count
      - max_findings / max_findings_items / max_findings_count
      - max_chars / max_content_chars / max_text_chars / max_bytes

    Behavior:
      - evidence/findings: slice lists to max items.
      - content/text: truncate to max_chars (or max_bytes treated as UTF-8 byte limit).

    Returns: (possibly-updated obj, did_trim, truncated_summary, error_code)
    """
    if not isinstance(obj, dict):
        return obj, False, None, None

    truncated: dict[str, Any] = {}

    max_e = _coerce_int(_first_present(budget, ["max_evidence", "max_evidence_items", "max_evidence_count"]))
    max_f = _coerce_int(_first_present(budget, ["max_findings", "max_findings_items", "max_findings_count"]))

    max_chars = _coerce_int(_first_present(budget, ["max_chars", "max_content_chars", "max_text_chars"]))
    max_bytes = _coerce_int(_first_present(budget, ["max_bytes"]))

    # If only max_bytes is present, treat it as a text cap for 'content'/'text' as well.
    if max_chars is None and max_bytes is not None:
        max_chars = max_bytes

    # Lists: evidence/findings
    try:
        if isinstance(max_e, int) and isinstance(obj.get("evidence"), list):
            before = len(obj["evidence"])
            if before > max_e:
                obj["evidence"] = obj["evidence"][:max_e]
                truncated["evidence"] = {"before": before, "after": len(obj["evidence"])}

        if isinstance(max_f, int) and isinstance(obj.get("findings"), list):
            before = len(obj["findings"])
            if before > max_f:
                obj["findings"] = obj["findings"][:max_f]
                truncated["findings"] = {"before": before, "after": len(obj["findings"])}
    except Exception:
        # Fail open: caller will record trimming_failed and continue
        return obj, False, None, "trim_lists_failed"

    # Text fields: content/text
    try:
        if isinstance(max_chars, int) and max_chars > 0:
            for field in ("content", "text"):
                v = obj.get(field)
                if isinstance(v, str):
                    before = len(v)
                    if before > max_chars:
                        obj[field] = v[:max_chars]
                        truncated[field] = {"chars_before": before, "chars_after": len(obj[field])}
    except Exception:
        return obj, False, None, "trim_text_failed"

    did_trim = bool(truncated)
    return obj, did_trim, (truncated if did_trim else None), None


def write_json_cache(
    project_root: str | Path,
    rel_path: str | Path,
    obj: Any,
    *,
    st: Optional[object] = None,
    stats: Optional[dict] = None,
    rec_id: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> Path:
    """
    Thin wrapper that stores JSON artifacts under a repository-local cache prefix
    (.aidev/cache/<rel_path>) and delegates to the central aidev.io_utils.write_json_cache
    implementation.

    This wrapper validates that the final target is repository-locked by calling
    aidev.io_utils._resolve_safe_path before delegating, and preserves metadata
    fields (st, stats, rec_id) by forwarding them to the underlying implementation.

    Additionally, if the object contains meta.budget with a merge_strategy of
    'deterministic', we attempt to read an existing cached artifact and merge
    the budget fields deterministically before writing. This supports incremental
    budget updates without nondeterministic overwrites.

    This wrapper also:
    - ensures a top-level schema_version is present (defaults to 1)
    - injects meta.ts (ISO8601 UTC, no subseconds) if missing; callers/tests should
      ignore meta.ts for determinism checks. meta.ts is the single volatile timestamp
      injected by this helper for reproducibility tests; callers should avoid adding
      additional volatile created_at fields unless explicitly required.

    The optional cache_key parameter allows callers to opt-in to deterministic,
    stable filenames for artifacts (preferred over rec_id when both are provided).

    Never mutates the caller-provided obj in-place.
    """
    project_root = Path(project_root)
    rel_path = Path(rel_path)

    # Ensure artifacts are placed under the repo-local cache prefix with a deterministic filename
    rel_cache = _deterministic_rel_cache(project_root, rel_path, rec_id=rec_id, cache_key=cache_key)

    # Validate/resolve using the central utility to enforce root-locked semantics
    # _resolve_safe_path will raise if rel_cache would escape project_root
    _resolve_safe_path(project_root, rel_cache)

    # Shallow-copy obj and inject schema_version/meta.ts if applicable (do not mutate caller object)
    if isinstance(obj, dict):
        obj = dict(obj)
        if "schema_version" not in obj:
            obj["schema_version"] = 1

        meta = obj.get("meta")
        if meta is None:
            meta = {}
        if isinstance(meta, dict):
            meta = dict(meta)
            if "ts" not in meta:
                import datetime

                # ISO8601 UTC with no subseconds and explicit Z suffix (e.g. 2025-12-19T12:34:56Z)
                meta["ts"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            obj["meta"] = meta

    # If incoming object contains meta.budget with deterministic merge_strategy,
    # attempt to merge with any existing cached artifact's budget before writing.
    try:
        if isinstance(obj, dict):
            meta = obj.get("meta")
            if isinstance(meta, dict):
                budget = meta.get("budget")
                if isinstance(budget, dict):
                    merge_strategy = budget.get("merge_strategy")
                    if merge_strategy == "deterministic":
                        merge_keys = budget.get("merge_keys")

                        # Read existing cached object (if any)
                        existing = io_read_json_cache(project_root, rel_cache)
                        if isinstance(existing, dict):
                            existing_meta = existing.get("meta")
                            if isinstance(existing_meta, dict):
                                existing_budget = existing_meta.get("budget")
                                if isinstance(existing_budget, dict):
                                    # Perform deterministic merge of the budgets using the public helper
                                    merged_budget = merge_meta_budgets(
                                        existing_budget,
                                        budget,
                                        merge_strategy="deterministic",
                                        merge_keys=merge_keys,
                                    )
                                    # Normalize merged budget for stable cache-key compatibility
                                    # (compute_cache_key uses normalize_budget_for_cache)
                                    merged_budget = normalize_budget_for_cache(merged_budget)

                                    # Create shallow copies to avoid mutating caller's data
                                    obj = dict(obj)
                                    obj_meta = dict(meta)
                                    obj_meta["budget"] = merged_budget
                                    obj["meta"] = obj_meta
                                    logger.debug(
                                        "Merged budget for %s using deterministic strategy; merge_keys=%s",
                                        rel_cache,
                                        merge_keys,
                                    )
    except Exception:
        # Be conservative: on any error during merging, fall back to writing incoming obj
        logger.exception("Error while attempting deterministic budget merge; writing incoming object as-is")

    # Enforce caps for oversized deep research artifacts based on meta.budget and record truncation.
    # This supports stable, predictable downstream LLM payload packing.
    # NOTE: budget key names are expected to be aligned with aidev/config.py; we support aliases:
    #   max_evidence/max_evidence_items, max_findings/max_findings_items, max_chars/max_bytes.
    did_trim = False
    try:
        if isinstance(obj, dict):
            meta = obj.get("meta")
            if isinstance(meta, dict):
                # Normalize/clamp budget before use and persist normalized form into obj['meta']['budget'].
                b = meta.get("budget")
                if isinstance(b, dict):
                    b_norm = normalize_budget_for_cache(b)
                    plan_caps = b_norm.get("plan_caps") if isinstance(b_norm.get("plan_caps"), dict) else None
                    b_norm = _clamp_budget_values(b_norm, plan_caps)
                    # Ensure normalized/clamped budget is stored for determinism across tools.
                    obj = dict(obj)
                    meta2 = dict(meta)
                    meta2["budget"] = b_norm

                    # Apply caps/truncation based on the normalized budget.
                    # Fail open: if trimming fails, record an error and still write.
                    obj2, did_trim, truncated_summary, err = _apply_artifact_caps(obj, b_norm)

                    if err is not None:
                        # Record trimming failure for observability; do not raise.
                        errs = meta2.get("errors")
                        if not isinstance(errs, list):
                            errs = []
                        errs = list(errs)
                        errs.append(f"trimming_failed:{err}")
                        meta2["errors"] = errs

                    if truncated_summary is not None:
                        meta2["truncated"] = truncated_summary

                    obj2["meta"] = meta2
                    obj = obj2
    except Exception:
        # Fail open: write untrimmed object but note trimming_failed for observability.
        logger.exception("Error while enforcing artifact caps; writing object as-is")
        try:
            if isinstance(obj, dict):
                meta = obj.get("meta")
                if isinstance(meta, dict):
                    obj = dict(obj)
                    meta2 = dict(meta)
                    errs = meta2.get("errors")
                    if not isinstance(errs, list):
                        errs = []
                    errs = list(errs)
                    errs.append("trimming_failed:exception")
                    meta2["errors"] = errs
                    obj["meta"] = meta2
        except Exception:
            pass

    # Delegate the actual write (including metadata handling) to the central implementation
    written_path = io_write_json_cache(project_root, rel_cache, obj, st=st, stats=stats, rec_id=rec_id)

    # Emit cache_write event (non-fatal). Payload must be small + deterministic.
    try:
        import json

        schema_version = obj.get("schema_version") if isinstance(obj, dict) else None
        meta = obj.get("meta") if isinstance(obj, dict) else None

        # Best-effort counts should reflect any trimming.
        size_bytes = None
        item_count = None
        if isinstance(stats, dict):
            size_bytes = stats.get("size_bytes")
            item_count = stats.get("item_count")

        # Recompute stats if we trimmed (or if missing), so events reflect the on-disk artifact.
        if did_trim or size_bytes is None or item_count is None:
            # Estimate without embedding raw contents.
            try:
                size_bytes = len(
                    json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )
            except Exception:
                size_bytes = 0

            # item_count: best-effort sum of common list lengths
            try:
                ic = 0
                if isinstance(obj, dict):
                    if isinstance(obj.get("evidence"), list):
                        ic += len(obj.get("evidence"))
                    if isinstance(obj.get("findings"), list):
                        ic += len(obj.get("findings"))
                item_count = ic
            except Exception:
                item_count = 0

        # Ensure stable, non-nullable integer fields for downstream consumers
        size_bytes_i = int(size_bytes) if isinstance(size_bytes, (int, float)) else 0
        item_count_i = int(item_count) if isinstance(item_count, (int, float)) else 0

        # When cache_key is explicitly provided (even if falsy), prefer it; otherwise fall back to rec_id
        chosen_cache_key = cache_key if cache_key is not None else (rec_id or "")

        payload = {
            "cache_key": str(chosen_cache_key),
            "artifact_digest": _extract_artifact_digest(rel_cache),
            "rel_cache_path": str(rel_cache),
            "schema_version": schema_version,
            "meta": _sanitize_meta_for_event(meta),
            "stats": {
                "size_bytes": size_bytes_i,
                "item_count": item_count_i,
            },
            "rec_id": rec_id,
        }
        # Use canonical, namespaced event type
        _emit_deep_research_event("deep_research.cache_write", payload)
    except Exception:
        logger.exception("Failed to prepare cache_write event payload")

    return written_path


def read_json_cache(project_root: str | Path, rel_path: str | Path, rec_id: Optional[str] = None, cache_key: Optional[str] = None) -> Optional[Any]:
    """
    Thin wrapper that maps rel_path into the repository-local cache prefix
    (.aidev/cache/<rel_path>) and delegates to aidev.io_utils.read_json_cache.

    Returns the parsed object on success, or None on miss/error (same semantics as the
    underlying implementation).

    Note: read_json_cache accepts an optional rec_id and cache_key so callers may read the same
    deterministic key that write_json_cache used when rec_id or cache_key were provided.
    When both are provided, cache_key is preferred for deterministic filename resolution.
    """
    project_root = Path(project_root)
    rel_path = Path(rel_path)

    rel_cache = _deterministic_rel_cache(project_root, rel_path, rec_id=rec_id, cache_key=cache_key)

    # Validate using the central utility to enforce root-locked semantics
    _resolve_safe_path(project_root, rel_cache)

    result = io_read_json_cache(project_root, rel_cache)

    # Emit cache_hit/cache_miss event (non-fatal). Do not include raw artifact content.
    try:
        schema_version = result.get("schema_version") if isinstance(result, dict) else None
        meta = result.get("meta") if isinstance(result, dict) else None

        # When cache_key is explicitly provided (even if falsy), prefer it; otherwise fall back to rec_id
        chosen_cache_key = cache_key if cache_key is not None else (rec_id or "")

        payload = {
            "cache_key": str(chosen_cache_key),
            "artifact_digest": _extract_artifact_digest(rel_cache),
            "rel_cache_path": str(rel_cache),
            "schema_version": schema_version,
            "meta": _sanitize_meta_for_event(meta),
            # Stats are best-effort; provide non-nullable ints to simplify downstream aggregation.
            "stats": {
                "size_bytes": 0,
                "item_count": 0,
            },
            "rec_id": rec_id,
        }

        if result is None:
            _emit_deep_research_event("deep_research.cache_miss", payload)
        else:
            _emit_deep_research_event("deep_research.cache_hit", payload)
    except Exception:
        logger.exception("Failed to prepare cache_hit/cache_miss event payload")

    return result


def get_cache_metadata(
    project_root: str | Path,
    rel_path: str | Path,
    *,
    request: Optional[dict] = None,
    repo_brief_hash: Optional[str] = None,
    rec_id: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> dict:
    """Read-only helper to surface deterministic cache metadata for DeepResearch.

    Contract: returns a dict with:
      - cache_key: str (best-effort; prefers explicit cache_key, else computed from request+repo_brief_hash, else rec_id, else '')
      - rel_cache_path: Path (repository-relative path under .aidev/cache)
      - budget: dict | None (from cached artifact meta.budget if present)
      - errors: list[str] (human-readable, non-fatal issues encountered)

    Notes:
      - This helper never raises; it logs exceptions and records them in errors.
      - Intended for deep_research_engine.py to populate ResearchBrief.meta (cache_key/budget/errors)
        without changing write_json_cache/read_json_cache behavior.
    """
    errors: list[str] = []
    project_root_p = Path(project_root)
    rel_path_p = Path(rel_path)

    computed_key: Optional[str] = None
    if request is not None and repo_brief_hash:
        try:
            # compute_cache_key will strip volatile meta and incorporate meta.profile/meta.budget if present
            computed_key = compute_cache_key(request, repo_brief_hash)
        except Exception as e:
            logger.exception("Failed to compute cache_key")
            errors.append(f"compute_cache_key failed: {e}")

    # Priority: explicit cache_key (when provided and possibly empty) > computed_key > rec_id
    seed_key = cache_key if cache_key is not None else (computed_key or rec_id)

    try:
        rel_cache = _deterministic_rel_cache(
            project_root_p,
            rel_path_p,
            rec_id=rec_id,
            cache_key=seed_key,
        )
        _resolve_safe_path(project_root_p, rel_cache)
    except Exception as e:
        logger.exception("Failed to resolve deterministic cache path")
        errors.append(f"resolve cache path failed: {e}")
        # Best-effort fallback (still repository-relative in the return payload)
        rel_cache = Path(".aidev") / "cache" / rel_path_p

    budget: Optional[dict] = None
    try:
        existing = io_read_json_cache(project_root_p, rel_cache)
        if isinstance(existing, dict):
            meta = existing.get("meta")
            if isinstance(meta, dict):
                b = meta.get("budget")
                if isinstance(b, dict):
                    budget = dict(b)
    except Exception as e:
        logger.exception("Failed to read/inspect cache artifact")
        errors.append(f"read cache failed: {e}")

    return {
        "cache_key": str(seed_key or ""),
        "rel_cache_path": rel_cache,
        "budget": budget,
        "errors": errors,
    }


def impact_surface_for_paths(
    project_root: str | Path,
    paths: list[str | Path],
    *,
    top_k: Optional[int] = 10,
    rec_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Best-effort helper that returns "scored dep_edge evidence" for the provided
    list of paths. This function produces deterministic, reproducible scores based
    on the project_root, the supplied rec_id, and the pair of paths themselves.

    This is intentionally a lightweight, local utility: it does not consult an
    external service. Instead it returns a list of evidence records of the form:
      {"from": <path>, "to": <path>, "score": <0.0-1.0 float>, "evidence": <str explaining score>}

    - paths: sequence of file paths (strings or Path) for which to compute impact edges.
    - top_k: if provided, limit the outgoing edges per source to the top_k highest-scoring targets.
    - rec_id: optional run identifier included in the deterministic score seed.

    The returned list is sorted by descending score.
    """
    project_root = str(project_root)

    # Normalize input paths to strings and unique list preserving order
    seen = set()
    norm_paths: list[str] = []
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            norm_paths.append(s)

    if len(norm_paths) < 2:
        return []

    records: list[dict[str, Any]] = []

    # For each ordered pair (src -> dst) compute a deterministic pseudo-score
    for src in norm_paths:
        candidates: list[tuple[str, float]] = []
        for dst in norm_paths:
            if dst == src:
                continue
            seed = f"{project_root}||{rec_id or ''}||{src}||{dst}"
            h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            # Use a portion of the hex digest to produce a stable float in [0, 1)
            intval = int(h[:16], 16)
            score = (intval % 1000000) / 1000000.0
            candidates.append((dst, score))

        # Sort candidates by descending score and keep top_k if requested
        candidates.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            candidates = candidates[:top_k]

        for dst, score in candidates:
            # Short deterministic seed for human-readable evidence and stable tiebreaking
            short_seed = hashlib.sha256((project_root + '|' + (rec_id or '') + '|' + src + '|' + dst).encode('utf-8')).hexdigest()[:12]
            why = f"Deterministic score derived from hash seed {short_seed}"
            records.append(
                {
                    "from": src,
                    "to": dst,
                    # Provide aliases for compatibility: some callers expect 'source'/'target'
                    "source": src,
                    "target": dst,
                    "score": float(score),
                    "kind": "dep_edge",
                    "why": why,
                    "evidence": why,
                }
            )

    # Global sort by score desc to give consumers a prioritized list
    # Use stable sort with tuple tiebreaker to ensure deterministic ordering
    records.sort(key=lambda r: (r["score"], r.get("from", ""), r.get("to", "")), reverse=True)
    return records

# fmt: on
