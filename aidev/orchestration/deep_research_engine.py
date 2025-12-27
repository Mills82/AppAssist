"""DeepResearchEngine skeleton: dataclasses and a safe, repo-root-locked run() stub.

Provides ResearchRequest, EvidenceItem, ResearchPlan, ResearchBrief dataclasses and
DeepResearchEngine with helpers for query normalization and deterministic cache-key
computation. No side-effects occur at import time; git/subprocess work is performed
inside run() and is best-effort. The run() implementation returns a placeholder
ResearchBrief whose meta.errors == ["not_implemented"] and llm_calls_used == 0.

This file was extended to include simple regex-based symbol indexing: text files
are scanned with lightweight language-heuristic regexes for symbol definitions
(e.g. Python def/class, JS function/const assignments). When symbols are found a
symbols key (list of name/line/context dicts) is added to the evidence dict and
such evidence is given kind == 'symbol_ref' by default. This supports downstream
symbol indexing and filtering while remaining deterministic and best-effort.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import unicodedata
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# Do not mutate sys.path at import time. Test harnesses or the runtime environment
# should ensure the repository root is on sys.path (for example via the test
# runner configuration or a sitecustomize.py). Import-time insertion of repo
# roots is brittle and was removed per repository guidelines.

# Import shared validators from the central helpers module. This module re-exports
# these names via __all__ so callers that import them from here continue to work.
try:
    from aidev.validators import validate_repo_schema, validate_research_plan, validate_research_brief
except Exception:
    # Best-effort import: consumers that need validation should ensure
    # aidev.validators is importable in the runtime environment. If import
    # fails, the rest of the module remains usable for non-validation paths.
    validate_repo_schema = None  # type: ignore
    validate_research_plan = None  # type: ignore
    validate_research_brief = None  # type: ignore

# Attempt to import deep_research_cache helper module when available. Tests
# that require the deep_research_cache contract can provide this module; if it
# is not present we fall back to an internal deterministic computation.
try:
    from aidev import deep_research_cache  # type: ignore
except Exception:
    deep_research_cache = None  # type: ignore

# Attempt to import authoritative bounded evidence builder when available. If not
# present we fall back to the local gather_basic_evidence implementation.
try:
    from aidev.orchestration.research import build_bounded_evidence  # type: ignore
except Exception:
    build_bounded_evidence = None  # type: ignore

# Attempt to import trace infrastructure when available.
try:
    from aidev import trace as _trace  # type: ignore
except Exception:
    _trace = None  # type: ignore

# Attempt to import an events bus when available.
try:
    from aidev import events_bus as _events_bus  # type: ignore
except Exception:
    _events_bus = None  # type: ignore


# Canonical event names
DEEP_RESEARCH_PHASE_STARTED = "deep_research.phase_started"
DEEP_RESEARCH_PHASE_DONE = "deep_research.phase_done"
DEEP_RESEARCH_CACHE_HIT = "deep_research.cache_hit"
DEEP_RESEARCH_CACHE_MISS = "deep_research.cache_miss"
DEEP_RESEARCH_ARTIFACT_WRITTEN = "deep_research.artifact_written"
DEEP_RESEARCH_BUDGET_UPDATE = "deep_research.budget_update"

# Canonical lifecycle events (high-level)
DEEP_RESEARCH_REQUEST_RECEIVED = "deep_research.request_received"
DEEP_RESEARCH_PLAN_CREATED = "deep_research.plan_created"
DEEP_RESEARCH_GATHER_STATS = "deep_research.gather_stats"
# Use fully namespaced canonical synthesize event string per validation feedback
DEEP_RESEARCH_SYNTH_DONE = "deep_research.synthesize_done"
DEEP_RESEARCH_VERIFY_DONE = "deep_research.verify_done"
DEEP_RESEARCH_VERIFY_SKIPPED = "deep_research.verify_skipped"


def _utc_now_iso_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_err_str(e: Exception) -> str:
    # Avoid tracebacks and keep it short; do not include any absolute paths.
    try:
        s = str(e)
    except Exception:
        s = e.__class__.__name__
    if not isinstance(s, str):
        s = e.__class__.__name__
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) > 240:
        s = s[:240] + "…"
    return s


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure payload is JSON-serializable and does not include oversized fields.

    This function intentionally avoids emitting large / sensitive fields:
    - drops 'evidence', 'repo_index' arrays when present
    - drops large strings under keys commonly used for excerpts/contents
    - ensures cache_ref/path-like fields are repo-relative (no absolute paths)
    """
    if not isinstance(payload, dict):
        return {}

    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if k in ("evidence", "repo_index"):
            continue
        if k in ("excerpt", "contents", "text"):
            continue
        if k in ("cache_ref", "cache_rel", "cache_path"):
            if isinstance(v, str) and v and not Path(v).is_absolute():
                out["cache_ref"] = v
            continue
        if k in ("path", "project_root", "abs_path"):
            # Never emit filesystem paths in events from this file.
            continue
        # Keep budgets reasonably bounded: copy as-is but strip nested objects if huge
        if k in ("before_budget", "after_budget", "budget", "caps", "actual") and isinstance(v, dict):
            # Keep only JSON-serializable primitives in these dicts
            bd: Dict[str, Any] = {}
            for bk, bv in v.items():
                if isinstance(bv, (str, int, float, bool)) or bv is None:
                    bd[bk] = bv
            out[k] = bd
            continue
        # Basic JSON-friendly pass-through
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            # Keep small lists of primitives only
            lst = []
            for item in v[:50]:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    lst.append(item)
            out[k] = lst
        elif isinstance(v, dict):
            # Shallow sanitize nested dicts
            nd: Dict[str, Any] = {}
            for nk, nv in v.items():
                if isinstance(nv, (str, int, float, bool)) or nv is None:
                    nd[nk] = nv
            out[k] = nd
        else:
            # Drop non-serializable types
            continue

    # Ensure timestamp exists to make consumption easier
    out.setdefault("ts", _utc_now_iso_z())
    return out


def _emit_deep_research_event(event_name: str, payload: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> None:
    """Best-effort event emission.

    Tries events bus emit methods if present; otherwise logs at DEBUG.
    Always sanitizes payload to avoid absolute paths and large contents.
    """
    lg = logger or logging.getLogger(__name__)
    safe_payload = _sanitize_payload(payload or {})

    # Try emitting to events bus
    try:
        bus = _events_bus
        if bus is not None:
            for method_name in ("emit", "emit_event", "publish"):
                meth = getattr(bus, method_name, None)
                if callable(meth):
                    try:
                        meth(event_name, safe_payload)
                        break
                    except Exception:
                        # Try next method
                        continue
    except Exception:
        pass

    # Try writing to trace infra (best-effort)
    try:
        tr = _trace
        if tr is not None:
            for method_name in ("record", "emit", "trace"):
                meth = getattr(tr, method_name, None)
                if callable(meth):
                    try:
                        meth(event_name, safe_payload)
                        break
                    except Exception:
                        continue
    except Exception:
        pass

    # Always allow debug logging fallback for visibility
    try:
        lg.debug("event %s: %s", event_name, safe_payload)
    except Exception:
        pass


# Public names exported by this module
__all__ = [
    "ResearchRequest",
    "EvidenceItem",
    "ResearchPlan",
    "ResearchBrief",
    "PreflightResult",
    "DeepResearchEngine",
    # Cache helpers
    "write_json_cache",
    "read_json_cache",
    # Budget merge helper
    "merge_meta_budgets",
    # Symbol helpers
    "index_symbols",
    "find_symbol_references_in_text",
    "extract_evidence_items",
    # Import/depgraph helpers
    "extract_imports_from_text",
    "build_dependency_graph",
    "impact_surface_for_paths",
    # Brief renderer
    "render_research_summary",
    # Public run helper (stable callable for analyze_mixin)
    "run_deep_research",
    "render_readable_summary",
    # Research context bundle
    "build_research_context_bundle",
    # Validation helpers
    "validate_repo_schema",
    "validate_research_plan",
]

# Canonical evidence id pattern used by the research brief schema
EVIDENCE_ID_REGEX = re.compile(r"^ev_[A-Za-z0-9_.-]+$")


def _ensure_path_within_root(project_root: Path, candidate: Path) -> Path:
    """Resolve candidate path and assert it remains under project_root.

    This mirrors DeepResearchEngine._ensure_within_project_root but is a
    module-level helper suitable for the top-level cache helpers below.
    """
    resolved_root = Path(project_root).resolve()
    resolved = Path(candidate).resolve()
    common = os.path.commonpath([str(resolved_root), str(resolved)])
    if common != str(resolved_root):
        raise ValueError(f"path {resolved} is outside of project root {resolved_root}")
    return resolved


def _canonical_ev_id(raw_id: Optional[str], rel: str, kind: str) -> str:
    """Return an evidence id that matches the canonical pattern ^ev_[A-Za-z0-9_.-]+$.

    Rules:
    - If raw_id already matches the pattern, return it unchanged.
    - If raw_id is present but contains disallowed chars, sanitize by replacing
      invalid characters with '_' and prefixing with 'ev_' if necessary.
    - Otherwise, deterministically derive an id using a sha256 of rel and kind
      and return 'ev_{hex}' (first 16 hex chars) which always matches the pattern.

    This keeps id generation deterministic and safe for downstream schema
    validation while preserving existing good ids.
    """
    if isinstance(raw_id, str) and EVIDENCE_ID_REGEX.match(raw_id):
        return raw_id
    if isinstance(raw_id, str) and raw_id:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_id)
        if not sanitized.startswith("ev_"):
            sanitized = "ev_" + sanitized
        if EVIDENCE_ID_REGEX.match(sanitized):
            return sanitized
    # Deterministic fallback using rel and kind
    seed = (rel or "") + "|" + (kind or "")
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"ev_{h}"


def _normalize_and_ensure_unique_ev_id(original_raw_id: Optional[str], rel: str, kind: str, seen: set, seed_fields: Optional[Dict[str, Any]] = None) -> (str, List[str]):
    """Normalize an ev_id to canonical form and ensure uniqueness within `seen`.

    Returns (ev_id, messages) where messages is a list of short string notes
    describing normalization or regeneration that callers should append to
    brief_meta.errors. Generation is deterministic based on rel/kind and
    seed_fields when duplication occurs.
    """
    msgs: List[str] = []
    ev_id = _canonical_ev_id(original_raw_id, rel or "", kind or "unknown")
    if isinstance(original_raw_id, str) and original_raw_id and original_raw_id != ev_id:
        msgs.append(f"ev_id_normalized:{original_raw_id}->{ev_id}")
    if not original_raw_id:
        msgs.append(f"ev_id_generated:{ev_id}")

    # Ensure uniqueness: if collision, deterministically derive a new id using a
    # sha256 over stable seed fields plus a counter until unique.
    counter = 0
    while ev_id in seen:
        seed_parts = [str(rel or ""), str(kind or "")]
        if seed_fields:
            seed_parts.append(str(seed_fields.get("line", "")))
            seed_parts.append(str(seed_fields.get("excerpt", "")))
            seed_parts.append(str(seed_fields.get("symbol", "")))
        seed_parts.append(str(counter))
        s = "|".join(seed_parts)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
        new_ev_id = f"ev_{h}"
        msgs.append(f"ev_id_collided:{ev_id}->{new_ev_id}")
        ev_id = new_ev_id
        counter += 1
    return ev_id, msgs


def write_json_cache(project_root, rel_path: str, obj: Any, *, st: Any = None, stats: Any = None, rec_id: Optional[str] = None) -> None:
    """Write a JSON artifact atomically under project_root/rel_path.

    - project_root: Path or path-like root directory for repository-scoped cache.
    - rel_path: repository-relative path (may include subdirectories); absolute
      paths are rejected for safety.
    - obj: JSON-serializable object to write. If st/stats/rec_id are provided they
      are included under a top-level "__meta__" key alongside the object in
      order to make writes auditable. The function performs an atomic write by
      writing to a temporary file in the target directory and then os.replace()'ing
      the temp file into place.

    Errors during write are propagated as exceptions to the caller to avoid
    silently losing cache writes.

    Note: the cache layout used by DeepResearchEngine is .aidev/cache/{cache_key}.json
    where cache_key is computed deterministically by DeepResearchEngine._compute_cache_key.
    The writer prefers a wrapper format {"__meta__": {...}, "value": obj} to
    make writes auditable; readers tolerate both raw dicts and this wrapped form.
    """
    root = Path(project_root)
    if Path(rel_path).is_absolute():
        raise ValueError("rel_path must be a repository-relative path")

    target_path = root.joinpath(rel_path)
    target_dir = target_path.parent
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Ensure the resolved path is within root to enforce root-locked policy.
    _ensure_path_within_root(root, target_path)

    # Prepare payload. If caller provided meta, put it under __meta__ to avoid
    # colliding with the user's top-level structure.
    payload = obj
    if any(v is not None for v in (st, stats, rec_id)):
        payload = {"__meta__": {"st": st, "stats": stats, "rec_id": rec_id}, "value": obj}

    # Serialize deterministically
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    # Write to temp file in same directory then atomically replace
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=str(target_dir), prefix=".tmp_") as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(serialized)
            tmp.flush()
            try:
                os.fsync(tmp.fileno())
            except Exception:
                # If fsync is not available on the platform, continue anyway;
                # the atomic replace still provides safety guarantees.
                pass
        # Final atomic move
        os.replace(str(tmp_path), str(target_path))
    finally:
        # If something went wrong and tmp exists, attempt to remove it
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def read_json_cache(project_root, rel_path: str) -> Optional[Any]:
    """Read a JSON artifact from project_root/rel_path.

    Returns the deserialized object on success or None if the file does not
    exist or cannot be read/parsed. Errors are logged at DEBUG level.

    The reader tolerates both raw dict payloads and the writer's wrapper
    format {"__meta__": {...}, "value": obj}.
    """
    logger = logging.getLogger(__name__)
    root = Path(project_root)
    if Path(rel_path).is_absolute():
        raise ValueError("rel_path must be a repository-relative path")

    target_path = root.joinpath(rel_path)

    # Ensure resolved path is within root
    try:
        _ensure_path_within_root(root, target_path)
    except Exception as e:
        logger.debug("read_json_cache: path check failed: %s", e)
        return None

    if not target_path.exists():
        return None

    try:
        with target_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.debug("read_json_cache: failed to read or parse %s: %s", target_path, e)
        return None


# Default budget values for the "standard" depth
DEFAULT_BUDGET = {
    "max_subquestions": 6,
    "max_evidence_items": 60,
    "max_files_touched": 6,
    "max_file_bytes": 40_000,
    "max_total_bytes": 160_000,
    "llm_calls_default": 3,
    "llm_calls_max": 5,
    # Phase-0 token budget bookkeeping
    "max_tokens": 2048,
}

# Named, explicit default budget profiles.
# Each profile is a complete dict with the same keys as DEFAULT_BUDGET.
DEFAULT_BUDGET_QUICK = {
    "max_subquestions": 3,
    "max_evidence_items": 12,
    "max_files_touched": 3,
    "max_file_bytes": 20_000,
    "max_total_bytes": 60_000,
    "llm_calls_default": 1,
    "llm_calls_max": 2,
    "max_tokens": 1024,
}

DEFAULT_BUDGET_STANDARD = DEFAULT_BUDGET.copy()

DEFAULT_BUDGET_DEEP = {
    "max_subquestions": 10,
    "max_evidence_items": 120,
    "max_files_touched": 12,
    "max_file_bytes": 80_000,
    "max_total_bytes": 400_000,
    "llm_calls_default": 5,
    "llm_calls_max": 8,
    "max_tokens": 4096,
}

BUDGET_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": DEFAULT_BUDGET_QUICK,
    "standard": DEFAULT_BUDGET_STANDARD,
    "deep": DEFAULT_BUDGET_DEEP,
}

# Hard caps enforced regardless of requested budgets. This guarantees bounded work
# even if callers request very large budgets.
HARD_CAPS: Dict[str, int] = {
    "max_subquestions": 20,
    "max_evidence_items": 200,
    "max_files_touched": 25,
    "max_file_bytes": 120_000,
    "max_total_bytes": 600_000,
    "llm_calls_default": 10,
    "llm_calls_max": 12,
    "max_tokens": 8192,
    # symbol indexing caps
    "per_symbol_limit": 8,
}


def resolve_budget(profile: Optional[str], provided_budgets: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve a named budget profile + provided overrides into a fully specified budget.

    - Starts from a named profile default (quick/standard/deep). Unknown profiles
      fall back to 'standard'.
    - Deterministically merges numeric keys using merge_meta_budgets(..., merge_strategy='deterministic').
    - Enforces HARD_CAPS by clamping numeric values and ensures required keys exist.
    - Ensures token bookkeeping keys exist: max_tokens, tokens_used, tokens_remaining.
    """
    prof = profile if isinstance(profile, str) and profile in BUDGET_PROFILES else "standard"
    base = dict(BUDGET_PROFILES.get(prof, DEFAULT_BUDGET_STANDARD))
    provided = dict(provided_budgets) if isinstance(provided_budgets, dict) else {}

    merge_keys = [
        "max_subquestions",
        "max_evidence_items",
        "max_files_touched",
        "max_file_bytes",
        "max_total_bytes",
        "llm_calls_default",
        "llm_calls_max",
        "max_tokens",
        "per_symbol_limit",
    ]

    # Ensure per_symbol_limit has a deterministic default if not present in profile.
    base.setdefault("per_symbol_limit", 3)

    try:
        merged = merge_meta_budgets(base, provided, merge_strategy="deterministic", merge_keys=merge_keys)
    except Exception:
        merged = dict(base)
        # best-effort apply provided directly for non-numeric values
        for k, v in provided.items():
            if not _is_number(v):
                merged[k] = v

    # Fully populate required keys (fallback to DEFAULT_BUDGET)
    for k, v in DEFAULT_BUDGET.items():
        merged.setdefault(k, v)
    merged.setdefault("per_symbol_limit", 3)

    # Clamp numeric values and enforce hard caps
    for k, cap in HARD_CAPS.items():
        try:
            if _is_number(merged.get(k)):
                merged[k] = int(max(0, min(int(merged.get(k)), int(cap))))
            elif k in ("per_symbol_limit",):
                # Ensure deterministic numeric default for this key
                merged[k] = int(max(0, min(int(merged.get(k) or 0), int(cap))))
        except Exception:
            # Fall back to cap for known numeric knobs
            merged[k] = int(max(0, int(cap)))

    # Ensure llm_calls_default <= llm_calls_max
    try:
        if _is_number(merged.get("llm_calls_default")) and _is_number(merged.get("llm_calls_max")):
            merged["llm_calls_default"] = min(int(merged["llm_calls_default"]), int(merged["llm_calls_max"]))
    except Exception:
        pass

    # Token bookkeeping
    max_tokens = merged.get("max_tokens")
    if not _is_number(max_tokens):
        max_tokens = DEFAULT_BUDGET.get("max_tokens", 0)
    max_tokens = int(max(0, int(max_tokens)))
    merged["max_tokens"] = max_tokens
    # Never carry tokens_used/tokens_remaining from caller; always deterministic initial values.
    merged["tokens_used"] = 0
    merged["tokens_remaining"] = max(max_tokens - 0, 0)

    return merged


def _is_number(v: Any) -> bool:
    # Exclude booleans (subclass of int)
    return (isinstance(v, (int, float)) and not isinstance(v, bool))


def _merge_budgets_deterministic(plan_budget: Dict[str, Any], cached_budget: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Deterministically merge two budget dicts for the provided keys.

    Strategy used here for 'deterministic' merge (updated):
    - For each key in keys, if both plan_budget and cached_budget provide a
      numeric value (int or float), choose the larger value (max) to allow reuse
      of previously-allocated allowances while remaining deterministic.
    - If only one side provides a numeric value, use that value.
    - Clamp resulting numeric values to be >= 0.
    - If the plan_budget includes an explicit cap for the field (either as
      '<key>_cap' or in a 'caps' dict), ensure the merged value does not
      exceed that cap.
    - For keys not listed, preserve the plan_budget value (prefer current plan).
    """
    merged = dict(plan_budget or {})
    if not isinstance(cached_budget, dict):
        # Ensure all numeric values are clamped to >= 0 even when not merging
        for k, v in list(merged.items()):
            if _is_number(v):
                merged[k] = max(0, v)
        return merged

    caps = {}
    if isinstance(plan_budget, dict):
        # gather caps if present under 'caps' dict
        caps = plan_budget.get("caps") if isinstance(plan_budget.get("caps"), dict) else {}

    for k in (keys or []):
        a = plan_budget.get(k) if isinstance(plan_budget, dict) else None
        b = cached_budget.get(k)
        val = None
        if _is_number(a) and _is_number(b):
            # Choose the MAX (allow previously allocated budget to persist)
            val = a if a > b else b
        elif _is_number(a):
            val = a
        elif _is_number(b):
            val = b
        # else: leave as-is (keep plan value if non-numeric)

        # If plan provides an explicit cap for this field, honor it.
        plan_cap = None
        # check '<key>_cap'
        cap_key = f"{k}_cap"
        if isinstance(plan_budget, dict) and _is_number(plan_budget.get(cap_key)):
            plan_cap = plan_budget.get(cap_key)
        # check 'caps' dict
        if plan_cap is None and isinstance(caps, dict) and _is_number(caps.get(k)):
            plan_cap = caps.get(k)

        if _is_number(val):
            # Clamp to non-negative
            val = max(0, val)
            # Apply plan cap if present
            if _is_number(plan_cap):
                val = min(val, plan_cap)
            merged[k] = val
        else:
            # If we didn't compute a numeric merged value, keep existing plan value
            # (already copied into merged at start)
            pass
    return merged

def merge_meta_budgets(old: Dict[str, Any], new: Dict[str, Any], merge_strategy: Optional[str] = None, merge_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Public wrapper to merge budgets coming from cached metadata (old) and
    the current plan/request (new).

    Parameters are named to match test expectations: 'old' is the cached budget
    metadata, 'new' is the current plan/request budget. If merge_strategy ==
    'deterministic' and merge_keys is a list, the deterministic merge is used.
    Otherwise, the 'new' budget is preferred (i.e., override behavior).

    The returned dict is the effective budget to use. Numeric fields are
    clamped to be >= 0 and deterministic when merged.
    """
    if not isinstance(old, dict):
        old = {}
    if not isinstance(new, dict):
        new = {}

    if merge_strategy == "deterministic" and isinstance(merge_keys, list):
        return _merge_budgets_deterministic(new, old, merge_keys)
    # Default: prefer new (plan) budget, but ensure numeric clamping
    merged = dict(new)
    for k, v in list(merged.items()):
        if _is_number(v):
            merged[k] = max(0, v)
    return merged


@dataclass
class ResearchRequest:
    """Input request for DeepResearchEngine.run().

    Fields are intentionally minimal and typed according to the spec.
    """

    query: str
    scope: Literal["repo", "subtree", "targets"] = "repo"
    scope_paths: List[str] = field(default_factory=list)
    depth: Literal["quick", "standard", "deep"] = "standard"
    force_refresh: bool = False
    budgets: Optional[Dict[str, Any]] = None


@dataclass
class EvidenceItem:
    ev_id: str
    path: str
    kind: str
    lines: Optional[str] = None
    excerpt: Optional[str] = None
    why: str = ""
    score: Optional[float] = None


@dataclass
class ResearchPlan:
    # Minimal plan shape to match expected schema fields.
    # Provide deterministic defaults so any accidental omission of fields when
    # constructing a ResearchPlan still yields schema-shaped objects. Callers
    # should supply explicit values where available; these defaults are the
    # conservative, deterministic fallbacks required by the tightened schemas.
    query: str = ""
    normalized_query: str = ""
    scope: str = "repo"
    scope_paths: List[str] = field(default_factory=list)
    depth: str = "standard"
    subquestions: List[Dict[str, Any]] = field(default_factory=list)
    priority_targets: List[str] = field(default_factory=list)
    impact_hypotheses: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=lambda: DEFAULT_BUDGET.copy())
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchBrief:
    # Core fields required by acceptance criteria
    # Provide defaults so downstream consumers always see the required keys.
    query: str = ""
    subquestions: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    impact_surface: List[Dict[str, Any]] = field(default_factory=list)
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    """Phase 0 result object.

    Determinism contract:
    - For identical inputs (request + repo_version + project_root contents that
      affect repo_version), cache_key and run_id are stable.
    - The persisted artifact is deterministic except for the explicitly allowed
      created_at timestamp field.
    """

    schema_version: int
    run_id: str
    cache_refs: List[str]
    normalized_request: Dict[str, Any]
    budget: Dict[str, Any]
    repo_version: str
    repo_index: Optional[List[str]] = None
    created_at: str = ""


def _dict_to_research_brief(obj: Dict[str, Any]) -> ResearchBrief:
    """Convert a plain dict (from JSON cache) into a ResearchBrief instance.

    This is forgiving: missing fields are defaulted where possible. Evidence
    items are converted into EvidenceItem instances if they look like dicts.
    """
    if not isinstance(obj, dict):
        raise TypeError("cached research brief must be a dict")

    query = obj.get("query", "")
    subquestions = obj.get("subquestions", []) or []

    evs = []
    for ev in obj.get("evidence", []) or []:
        if isinstance(ev, dict):
            # Only pass known fields to EvidenceItem; let dataclass reject missing
            # required fields by raising if necessary.
            try:
                kind = ev.get("kind", ev.get("type", ""))
                # If the evidence dict contains a 'symbols' key we treat it as
                # a symbol reference evidence item unless kind is explicitly set.
                if not kind and isinstance(ev.get("symbols"), list) and ev.get("symbols"):
                    kind = "symbol_ref"
                path = ev.get("path", "")
                raw_id = ev.get("ev_id", ev.get("id", ""))
                ev_id = _canonical_ev_id(raw_id, path or "", kind or "unknown")
                ev_item = EvidenceItem(
                    ev_id=ev_id,
                    path=path,
                    kind=kind,
                    lines=ev.get("lines"),
                    excerpt=ev.get("excerpt"),
                    why=ev.get("why", ""),
                    score=ev.get("score"),
                )
            except Exception:
                # If conversion fails, skip this evidence item but continue
                continue
            evs.append(ev_item)

    findings = obj.get("findings", []) or []
    impact_surface = obj.get("impact_surface", []) or []
    suggested_actions = obj.get("suggested_actions", []) or []
    gaps = obj.get("gaps", []) or []
    meta = obj.get("meta", {}) or {}

    return ResearchBrief(
        query=query,
        subquestions=list(subquestions),
        evidence=evs,
        findings=list(findings),
        impact_surface=list(impact_surface),
        suggested_actions=list(suggested_actions),
        gaps=list(gaps),
        meta=dict(meta),
    )


def _dict_to_research_plan(obj: Dict[str, Any]) -> ResearchPlan:
    """Convert a plain dict (from JSON cache) into a ResearchPlan instance.

    This is forgiving and performs minimal defaulting.
    """
    if not isinstance(obj, dict):
        raise TypeError("cached research plan must be a dict")

    return ResearchPlan(
        query=obj.get("query", ""),
        normalized_query=obj.get("normalized_query", ""),
        scope=obj.get("scope", "repo"),
        scope_paths=list(obj.get("scope_paths", []) or []),
        depth=obj.get("depth", "standard"),
        subquestions=list(obj.get("subquestions", []) or []),
        priority_targets=list(obj.get("priority_targets", []) or []),
        impact_hypotheses=list(obj.get("impact_hypotheses", []) or []),
        budget=dict(obj.get("budget", {}) or {}),
        meta=dict(obj.get("meta", {}) or {}),
    )


def _light_validate_plan_dict(plan_dict: Dict[str, Any]) -> bool:
    """Lightweight structural check when validate_research_plan is unavailable."""
    if not isinstance(plan_dict, dict):
        return False
    required = ["query", "normalized_query", "scope", "budget", "subquestions"]
    for k in required:
        if k not in plan_dict:
            return False
    if not isinstance(plan_dict.get("budget"), dict):
        return False
    if not isinstance(plan_dict.get("subquestions"), list):
        return False
    return True


def _normalize_brief_subquestions(subq_list: Any) -> List[str]:
    """Normalize plan subquestions into a deterministic list of strings.

    Accepts list items that may be dicts (preferred), strings, or other objects.
    For dicts, prefers 'question', then 'id', then a stable JSON repr.
    Empty/whitespace-only entries are dropped.
    """
    out: List[str] = []
    for s in (subq_list or []):
        try:
            if isinstance(s, dict):
                v = s.get("question") or s.get("id")
                if not isinstance(v, str) or not v.strip():
                    v = json.dumps(s, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                v = v.strip()
            elif isinstance(s, str):
                v = s.strip()
            else:
                v = str(s).strip()
            if v:
                out.append(v)
        except Exception:
            continue
    return out


def _dedupe_evidence_items(evs: List[EvidenceItem]) -> List[EvidenceItem]:
    """Deduplicate EvidenceItem list by ev_id, preserving first-seen order."""
    seen = set()
    out: List[EvidenceItem] = []
    for e in (evs or []):
        try:
            ev_id = getattr(e, "ev_id", None)
            if not isinstance(ev_id, str) or not ev_id:
                continue
            if ev_id in seen:
                continue
            seen.add(ev_id)
            out.append(e)
        except Exception:
            continue
    return out


def _filter_invalid_evidence_refs(items: Any, ev_id_set: set, *, kind: str, errors: List[str]) -> Any:
    """Strip invalid evidence_refs in findings/suggested_actions (best-effort).

    Mutates dict items in-place by replacing evidence_refs with filtered list.
    Appends a machine-readable error string to errors when removals occur.

    Additionally, when evidence refs are removed this function preserves
    traceability by keeping a small copy of the original refs under
    'evidence_refs_original' (first 50 entries) and annotates the item with
    'evidence_refs_removed' (int) and 'evidence_refs_insufficient' (bool).
    """
    if not isinstance(items, list) or not ev_id_set:
        return items
    removed_total = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        refs = it.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            continue
        kept = []
        removed = 0
        for r in refs:
            if isinstance(r, str) and r in ev_id_set:
                kept.append(r)
            else:
                removed += 1
        if removed:
            # Preserve a small snapshot of the original refs for traceability
            try:
                it["evidence_refs_original"] = [str(x) for x in (refs[:50] if isinstance(refs, list) else [])]
            except Exception:
                # be defensive; if serialization fails, omit original copy
                pass
            it["evidence_refs"] = kept
            it["evidence_refs_removed"] = removed
            it["evidence_refs_insufficient"] = True
            removed_total += removed
    if removed_total:
        errors.append(f"stripped_invalid_evidence_refs_in_{kind}:{removed_total}")
    return items


def _is_safe_relpath_str(p: Any) -> bool:
    """Return True iff p looks like a safe repo-relative path string."""
    if not isinstance(p, str) or not p.strip():
        return False
    try:
        if Path(p).is_absolute():
            return False
    except Exception:
        return False
    return True


def _format_line_ref(value: Any) -> Optional[str]:
    """Normalize a line reference into 'start-end' or 'line' string.

    Accepts:
    - string like '10-20' or '15'
    - int-like

    Returns None if value is not usable.
    """
    if value is None:
        return None
    if _is_number(value):
        try:
            n = int(value)
        except Exception:
            return None
        if n < 1:
            return None
        return str(n)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # allow patterns like '10-20' or '15'
        m = re.match(r"^(\d+)(?:\s*-\s*(\d+))?$", s)
        if not m:
            return None
        a = int(m.group(1))
        if a < 1:
            return None
        b = m.group(2)
        if b is None:
            return str(a)
        bb = int(b)
        if bb < a:
            bb = a
        return f"{a}-{bb}"
    return None


def render_research_summary(brief: Union[ResearchBrief, Dict[str, Any], None], *, max_findings: int = 5) -> Optional[str]:
    """Return a compact, evidence-cited string summary of a ResearchBrief.

    This is a pure, fail-safe helper intended for analyze/recommendation flows.
    It returns None for invalid input and never raises.

    Citations are formatted as:
      - "path" when no line info is available
      - "path:line" when a single line is available
      - "path:start-end" when a range is available
    """
    try:
        if isinstance(brief, dict):
            brief_obj = _dict_to_research_brief(brief)
        elif isinstance(brief, ResearchBrief):
            brief_obj = brief
        else:
            return None

        # Clamp max_findings deterministically
        try:
            mf = int(max_findings)
        except Exception:
            mf = 5
        if mf < 0:
            mf = 0

        # Build ev_id -> citation map
        citations: Dict[str, str] = {}
        evidence_seq = getattr(brief_obj, "evidence", []) or []
        for ev in evidence_seq:
            try:
                if isinstance(ev, EvidenceItem):
                    ev_id = getattr(ev, "ev_id", None)
                    path = getattr(ev, "path", None)
                    lines = getattr(ev, "lines", None)
                    line = None
                elif isinstance(ev, dict):
                    ev_id = ev.get("ev_id") or ev.get("id")
                    path = ev.get("path")
                    lines = ev.get("lines")
                    line = ev.get("line")
                else:
                    continue

                if not isinstance(ev_id, str) or not ev_id:
                    continue
                if not _is_safe_relpath_str(path):
                    # drop abs/invalid paths; keep mapping only for safe paths
                    continue

                line_ref = _format_line_ref(lines)
                if line_ref is None:
                    line_ref = _format_line_ref(line)

                if line_ref:
                    cite = f"{path}:{line_ref}"
                else:
                    cite = f"{path}"

                # Preserve first occurrence deterministically
                citations.setdefault(ev_id, cite)
            except Exception:
                continue

        query = getattr(brief_obj, "query", "") or ""
        meta = getattr(brief_obj, "meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        repo_version = meta.get("repo_version") if isinstance(meta.get("repo_version"), str) else None
        cache_key = meta.get("cache_key") if isinstance(meta.get("cache_key"), str) else None
        llm_calls_used = meta.get("llm_calls_used")
        llm_calls_s = None
        if _is_number(llm_calls_used):
            llm_calls_s = str(int(llm_calls_used))

        header_parts = ["Deep research summary"]
        if query:
            header_parts.append(f"query={query}")
        if repo_version:
            header_parts.append(f"repo_version={repo_version}")
        if cache_key:
            header_parts.append(f"cache_key={cache_key}")
        if llm_calls_s is not None:
            header_parts.append(f"llm_calls_used={llm_calls_s}")
        header = " | ".join(header_parts)

        # Render findings if present; else fall back to top evidence list
        findings = getattr(brief_obj, "findings", None)
        lines_out: List[str] = [header]

        if isinstance(findings, list) and findings:
            count = 0
            for i, f in enumerate(findings):
                if mf and count >= mf:
                    break
                if not isinstance(f, dict):
                    continue

                title = None
                for k in ("title", "summary", "finding"):
                    v = f.get(k)
                    if isinstance(v, str) and v.strip():
                        title = v.strip()
                        break
                if not title:
                    why = f.get("why")
                    if isinstance(why, str) and why.strip():
                        title = why.strip()[:120]
                    else:
                        title = f"Finding #{i+1}"

                refs = f.get("evidence_refs")
                cite_list: List[str] = []
                if isinstance(refs, list):
                    for r in refs:
                        if isinstance(r, str) and r in citations:
                            cite_list.append(citations[r])
                # Deterministic de-dupe, preserve order
                seen_c = set()
                cite_list_dedup = []
                for c in cite_list:
                    if c in seen_c:
                        continue
                    seen_c.add(c)
                    cite_list_dedup.append(c)

                if cite_list_dedup:
                    lines_out.append(f"- {title} (evidence: {', '.join(cite_list_dedup)})")
                else:
                    lines_out.append(f"- {title}")
                count += 1

            if mf and len(findings) > mf:
                lines_out.append(f"(truncated findings: showing {mf} of {len(findings)})")
        else:
            # Fallback: list up to 5 citations from evidence, deterministically in evidence order
            shown = 0
            for ev in evidence_seq:
                if shown >= 5:
                    break
                try:
                    if isinstance(ev, EvidenceItem):
                        ev_id = getattr(ev, "ev_id", None)
                    elif isinstance(ev, dict):
                        ev_id = ev.get("ev_id") or ev.get("id")
                    else:
                        continue
                    if isinstance(ev_id, str) and ev_id in citations:
                        lines_out.append(f"- evidence: {citations[ev_id]}")
                        shown += 1
                except Exception:
                    continue

        return "\n".join(lines_out).strip() if lines_out else None
    except Exception:
        return None


def run_deep_research(focus: Any, cards: Any, depth: Any) -> Dict[str, Any]:
    """Stable, top-level helper to run deep research.

    Signature required by analyze_mixin: run_deep_research(focus, cards, depth)
    Returns a dict with 'ok': bool. On success include a ResearchBrief-like
    dict under the 'brief' key (or 'research_brief').

    This implementation is intentionally defensive and best-effort:
    - If a proper DeepResearchEngine implementation is available in this
      module (DeepResearchEngine class), it will be instantiated and used.
    - If a higher-level builder (build_bounded_evidence) is available it may be
    used to assemble an evidence list.
    - Otherwise, the function returns ok==False and a lightweight placeholder
      brief (meta.errors=['not_implemented']) to keep callers fail-safe.
    """
    try:
        depth_s = str(depth) if depth is not None else "standard"
        # Prefer an in-module DeepResearchEngine if present
        engine_cls = globals().get("DeepResearchEngine")
        if engine_cls is not None:
            try:
                # Ensure we provide a project_root to avoid TypeError
                engine = engine_cls(Path(".").resolve())  # type: ignore
                # Try to prefer running with a ResearchRequest if the engine supports it
                req = ResearchRequest(query=str(focus) if focus is not None else "", depth=depth_s)
                res = engine.run(req)  # type: ignore
                # Normalize result
                if isinstance(res, ResearchBrief):
                    brief_obj = dataclasses.asdict(res)
                elif isinstance(res, dict):
                    brief_obj = res
                else:
                    brief_obj = None
                if brief_obj:
                    return {"ok": True, "brief": brief_obj}
                return {"ok": False, "error": "engine_returned_no_brief"}
            except Exception as e:
                return {"ok": False, "error": _safe_err_str(e)}

        # Best-effort lightweight brief from available 'cards' input when no engine
        brief: Dict[str, Any] = {"query": str(focus) if focus is not None else "", "subquestions": [], "evidence": [], "findings": [], "meta": {"errors": ["not_implemented"], "llm_calls_used": 0}}

        # If a build_bounded_evidence helper is available, use it to populate evidence
        try:
            if build_bounded_evidence is not None and isinstance(cards, (list, dict)):
                try:
                    evs = build_bounded_evidence(cards)
                    if isinstance(evs, list) and evs:
                        brief["evidence"] = evs
                except Exception:
                    # ignore and continue
                    pass
        except Exception:
            pass

        # If cards looks like a list of dicts with 'title' or 'name', use them as findings
        try:
            if isinstance(cards, list):
                for i, c in enumerate(cards[:5]):
                    if isinstance(c, dict):
                        title = c.get("title") or c.get("name") or c.get("text") or f"Card {i+1}"
                    else:
                        title = str(c)[:120]
                    brief["findings"].append({"title": title, "evidence_refs": []})
        except Exception:
            pass

        # Consider this a success only if we produced at least one finding or evidence
        if brief.get("findings") or brief.get("evidence"):
            # clear not_implemented error if we actually have content
            meta = brief.get("meta") or {}
            errs = [e for e in meta.get("errors", []) if e != "not_implemented"]
            meta["errors"] = errs
            brief["meta"] = meta
            return {"ok": True, "brief": brief}

        return {"ok": False, "brief": brief}
    except Exception as e:
        return {"ok": False, "error": _safe_err_str(e)}


def render_readable_summary(research_res: Any, *, max_findings: int = 5) -> Optional[str]:
    """Render a readable summary from a run_deep_research result.

    This adaptor prefers the 'brief' key when present, otherwise falls back
    to trying the provided value directly. It delegates to render_research_summary
    which is the canonical renderer for citation formatting.
    """
    try:
        if isinstance(research_res, dict) and research_res.get("brief"):
            brief = research_res.get("brief")
        else:
            brief = research_res
        return render_research_summary(brief, max_findings=max_findings)
    except Exception:
        return None


def build_research_context_bundle(
    brief: Union[ResearchBrief, Dict[str, Any], None],
    *,
    caps: Optional[Dict[str, Any]] = None,
    per_item_char_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a compact, deterministic, JSON-serializable research context bundle.

    Pure transformation only (no I/O, no randomness). Intended to be embedded into
    downstream LLM prompts when available.

    Output sections are always present in stable order:
      ["metadata", "top_findings", "evidence", "impact_surface", "gaps", "suggested_actions"].

    Deterministic bytes:
      json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    Args:
      brief: ResearchBrief or dict (will be normalized via _dict_to_research_brief).
      caps: optional caps dict. Supported forms:
        - flat: {"max_items": int, "max_chars": int}
        - per-section: {"evidence": {"max_items": int, "max_chars": int}, ...}
      per_item_char_limit: optional per-field character cap for text fields like
        excerpt/why/summary. When applied, per-item flags are included:
        <field>_truncated (bool), <field>_orig_len (int).

    Returns:
      dict with keys: sections, section_meta, meta.
    """

    def _as_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        if _is_number(v):
            try:
                i = int(v)
            except Exception:
                return None
            return i if i >= 0 else 0
        return None

    def _get_section_caps(section: str) -> Dict[str, Optional[int]]:
        # Accept a flat dict (applies to all sections) and/or per-section overrides.
        c = caps if isinstance(caps, dict) else {}
        flat_max_items = _as_int(c.get("max_items"))
        flat_max_chars = _as_int(c.get("max_chars"))
        sec = c.get(section) if isinstance(c.get(section), dict) else {}
        return {
            "max_items": _as_int(sec.get("max_items")) if isinstance(sec, dict) else None,
            "max_chars": _as_int(sec.get("max_chars")) if isinstance(sec.get(section) if isinstance(sec, dict) else sec, dict) else sc if False else (_as_int(sec.get("max_chars")) if isinstance(sec, dict) else None),
            "flat_max_items": flat_max_items,
            "flat_max_chars": flat_max_chars,
        }

    # The above attempt to be clever caused complexity; simplify _get_section_caps
    def _get_section_caps(section: str) -> Dict[str, Optional[int]]:
        c = caps if isinstance(caps, dict) else {}
        flat_max_items = _as_int(c.get("max_items"))
        flat_max_chars = _as_int(c.get("max_chars"))
        sec = c.get(section) if isinstance(c.get(section), dict) else {}
        return {
            "max_items": _as_int(sec.get("max_items")) if isinstance(sec, dict) else None,
            "max_chars": _as_int(sec.get("max_chars")) if isinstance(sec, dict) else None,
            "flat_max_items": flat_max_items,
            "flat_max_chars": flat_max_chars,
        }

    def _effective_caps_for_section(section: str) -> Dict[str, Optional[int]]:
        sc = _get_section_caps(section)
        # Per-section overrides win; fallback to flat.
        max_items = sc.get("max_items")
        if max_items is None:
            max_items = sc.get("flat_max_items")
        max_chars = sc.get("max_chars")
        if max_chars is None:
            max_chars = sc.get("flat_max_chars")
        return {"max_items": max_items, "max_chars": max_chars}

    def _truncate_text_field(obj: Dict[str, Any], field_name: str, limit: Optional[int]) -> None:
        if limit is None:
            return
        try:
            lim = int(limit)
        except Exception:
            return
        if lim < 0:
            return
        v = obj.get(field_name)
        if not isinstance(v, str):
            return
        orig_len = len(v)
        if orig_len <= lim:
            obj[f"{field_name}_truncated"] = False
            obj[f"{field_name}_orig_len"] = orig_len
            return
        obj[field_name] = v[:lim]
        obj[f"{field_name}_truncated"] = True
        obj[f"{field_name}_orig_len"] = orig_len

    def _apply_char_budget_to_items(items: List[Dict[str, Any]], *, budget: Optional[int], fields: List[str]) -> (List[Dict[str, Any]], Dict[str, Any]):
        # Deterministic: process in order, clamp each text field based on remaining budget.
        meta = {"truncated": False, "truncation_reasons": [], "truncation_point": None}
        if budget is None:
            return items, meta
        try:
            remaining = int(budget)
        except Exception:
            return items, meta
        if remaining < 0:
            return items, meta

        out: List[Dict[str, Any]] = []
        truncated_at: Optional[int] = None
        did_truncate = False
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            if remaining <= 0:
                truncated_at = idx
                did_truncate = True
                break
            d = dict(it)
            # Truncate relevant fields to fit in remaining budget.
            for f in fields:
                if remaining <= 0:
                    break
                v = d.get(f)
                if not isinstance(v, str):
                    continue
                orig_len = len(v)
                take = min(orig_len, remaining)
                if take < orig_len:
                    d[f] = v[:take]
                    d[f"{f}_truncated"] = True
                    d[f"{f}_orig_len"] = orig_len
                    did_truncate = True
                else:
                    d[f"{f}_truncated"] = False
                    d[f"{f}_orig_len"] = orig_len
                remaining -= take
            out.append(d)

        if did_truncate:
            meta["truncated"] = True
            meta["truncation_reasons"] = ["max_chars"]
            meta["truncation_point"] = truncated_at if truncated_at is not None else len(out)
        return out, meta

    def _cap_items_deterministic(items: List[Dict[str, Any]], max_items: Optional[int]) -> (List[Dict[str, Any]], Dict[str, Any]):
        meta = {"truncated": False, "truncation_reasons": [], "truncation_point": None}
        if max_items is None:
            return items, meta
        try:
            mi = int(max_items)
        except Exception:
            return items, meta
        if mi < 0:
            return items, meta
        if len(items) <= mi:
            return items, meta
        meta["truncated"] = True
        meta["truncation_reasons"] = ["max_items"]
        meta["truncation_point"] = mi
        return items[:mi], meta

    # Normalize input
    try:
        if isinstance(brief, ResearchBrief):
            b = brief
        elif isinstance(brief, dict):
            b = _dict_to_research_brief(brief)
        else:
            b = ResearchBrief(query="", subquestions=[], evidence=[], findings=[], impact_surface=[], suggested_actions=[], gaps=[], meta={})
    except Exception:
        b = ResearchBrief(query="", subquestions=[], evidence=[], findings=[], impact_surface=[], suggested_actions=[], gaps=[], meta={})

    # Fixed section order; always include keys even if empty.
    section_order = ["metadata", "top_findings", "evidence", "impact_surface", "gaps", "suggested_actions"]

    # Base metadata (keep small and deterministic)
    meta_src = b.meta if isinstance(b.meta, dict) else {}
    metadata_section: Dict[str, Any] = {
        "query": b.query if isinstance(b.query, str) else "",
        "repo_version": meta_src.get("repo_version") if isinstance(meta_src.get("repo_version"), str) else None,
        "cache_key": meta_src.get("cache_key") if isinstance(meta_src.get("cache_key"), str) else None,
        "llm_calls_used": int(meta_src.get("llm_calls_used", 0) or 0) if _is_number(meta_src.get("llm_calls_used", 0) or 0) else 0,
        "errors": list(meta_src.get("errors", []) or []) if isinstance(meta_src.get("errors", []), list) else [],
    }

    # Build top_findings: preserve input order but normalize fields deterministically.
    raw_findings = list(b.findings or []) if isinstance(b.findings, list) else []
    findings_items: List[Dict[str, Any]] = []
    for i, f in enumerate(raw_findings):
        if not isinstance(f, dict):
            continue
        # Prefer 'title', then 'summary', then 'finding', then 'why'.
        title = None
        for k in ("title", "summary", "finding"):
            v = f.get(k)
            if isinstance(v, str) and v.strip():
                title = v.strip()
                break
        if not title:
            why = f.get("why")
            if isinstance(why, str) and why.strip():
                title = why.strip()
        if not title:
            # stable fallback
            title = f"Finding #{i+1}"

        refs = f.get("evidence_refs")
        refs_out: List[str] = []
        if isinstance(refs, list):
            for r in refs:
                if isinstance(r, str) and r:
                    refs_out.append(r)
        # deterministic de-dupe preserving order
        seen_r = set()
        refs_dedup: List[str] = []
        for r in refs_out:
            if r in seen_r:
                continue
            seen_r.add(r)
            refs_dedup.append(r)

        item: Dict[str, Any] = {
            "title": title,
            "evidence_refs": refs_dedup,
            "confidence": f.get("confidence") if isinstance(f.get("confidence"), str) else None,
            "why": f.get("why") if isinstance(f.get("why"), str) else None,
            "__idx": i,
        }
        # Apply per-item truncation
        if per_item_char_limit is not None:
            _truncate_text_field(item, "title", per_item_char_limit)
            _truncate_text_field(item, "why", per_item_char_limit)
        findings_items.append(item)

    # Deterministically sort findings to satisfy acceptance criterion 1.
    try:
        findings_items_sorted = sorted(
            findings_items,
            key=lambda it: (
                (it.get("title") or "").casefold(),
                (it.get("evidence_refs")[0] if it.get("evidence_refs") else ""),
                int(it.get("__idx", 0)),
            ),
        )
    except Exception:
        findings_items_sorted = list(findings_items)

    # Remove internal index marker before including in output
    for it in findings_items_sorted:
        if "__idx" in it:
            del it["__idx"]

    # Evidence: preserve ev_id, do not synthesize/replace.
    evidence_items: List[Dict[str, Any]] = []
    raw_evidence = list(getattr(b, "evidence", []) or [])
    for ev in raw_evidence:
        if isinstance(ev, EvidenceItem):
            ev_dict = {
                "ev_id": ev.ev_id,
                "path": ev.path,
                "kind": ev.kind,
                "line": None,
                "lines": ev.lines,
                "score": ev.score,
                "why": ev.why if isinstance(ev.why, str) else "",
                "excerpt": ev.excerpt,
                # symbols are optional; EvidenceItem doesn't carry them
                "symbols": None,
            }
        elif isinstance(ev, dict):
            # Must preserve ev_id if present; do not replace.
            ev_id_val = ev.get("ev_id") if isinstance(ev.get("ev_id"), str) else (ev.get("id") if isinstance(ev.get("id"), str) else "")
            if not ev_id_val:
                # If missing, we still include evidence but with empty ev_id; caller can decide.
                ev_id_val = ""
            path_val = ev.get("path") if isinstance(ev.get("path"), str) else ""
            kind_val = ev.get("kind") if isinstance(ev.get("kind"), str) else (ev.get("type") if isinstance(ev.get("type"), str) else "unknown")
            line_val = ev.get("line")
            line_int = int(line_val) if _is_number(line_val) else None
            if line_int is not None and line_int < 1:
                line_int = None
            score_val = ev.get("score")
            score_f = float(score_val) if _is_number(score_val) else None

            symbols_val = ev.get("symbols")
            # Keep symbols as-is if list; else None.
            if isinstance(symbols_val, list):
                # Keep small list of strings/dicts only; deterministic truncation.
                norm_syms: List[Any] = []
                for s in symbols_val[:50]:
                    if isinstance(s, (str, int, float, bool)) or s is None:
                        norm_syms.append(s)
                    elif isinstance(s, dict):
                        # keep shallow dict
                        sd: Dict[str, Any] = {}
                        for sk, sv in s.items():
                            if isinstance(sv, (str, int, float, bool)) or sv is None:
                                sd[sk] = sv
                        norm_syms.append(sd)
                symbols_val = norm_syms
            else:
                symbols_val = None

            ev_dict = {
                "ev_id": ev_id_val,
                "path": path_val,
                "kind": kind_val,
                "line": line_int,
                "lines": ev.get("lines") if isinstance(ev.get("lines"), str) else (ev.get("line") if isinstance(ev.get("line"), str) else None),
                "score": score_f,
                "why": ev.get("why") if isinstance(ev.get("why"), str) else "",
                "excerpt": ev.get("excerpt") if isinstance(ev.get("excerpt"), str) else None,
                "symbols": symbols_val,
            }
        else:
            continue

        # Apply per-item truncation
        if per_item_char_limit is not None:
            _truncate_text_field(ev_dict, "why", per_item_char_limit)
            _truncate_text_field(ev_dict, "excerpt", per_item_char_limit)
        evidence_items.append(ev_dict)

    # Deterministic evidence ordering for bundle
    def _ev_score(v: Dict[str, Any]) -> float:
        try:
            sc = v.get("score")
            return float(sc) if _is_number(sc) else 0.0
        except Exception:
            return 0.0

    evidence_items_sorted = sorted(
        evidence_items,
        key=lambda e: (-_ev_score(e), e.get("path") or "", e.get("ev_id") or ""),
    )

    # impact_surface and suggested_actions/gaps: preserve order, but normalize+truncate text
    def _norm_list_of_dicts(src: Any, *, fields_to_truncate: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(src, list):
            return out
        for idx, it in enumerate(src):
            if not isinstance(it, dict):
                continue
            d = dict(it)
            # Normalize evidence_refs if present: list of strings, de-duped deterministically
            if isinstance(d.get("evidence_refs"), list):
                refs = [r for r in d.get("evidence_refs") if isinstance(r, str) and r]
                seen = set()
                dedup = []
                for r in refs:
                    if r in seen:
                        continue
                    seen.add(r)
                    dedup.append(r)
                d["evidence_refs"] = dedup
            if per_item_char_limit is not None:
                for f in fields_to_truncate:
                    _truncate_text_field(d, f, per_item_char_limit)
            # record index for deterministic sorting later
            d["__idx"] = idx
            out.append(d)
        return out

    impact_items = _norm_list_of_dicts(getattr(b, "impact_surface", None), fields_to_truncate=["why", "summary", "title"])
    gaps_items = _norm_list_of_dicts(getattr(b, "gaps", None), fields_to_truncate=["gap", "why", "next_query", "summary"])
    actions_items = _norm_list_of_dicts(getattr(b, "suggested_actions", None), fields_to_truncate=["title", "summary", "why"])

    # Deterministically sort impact/gaps/actions to guarantee stable output
    def _det_sort(lst: List[Dict[str, Any]], prim_fields: List[str]) -> List[Dict[str, Any]]:
        try:
            return sorted(
                lst,
                key=lambda it: (
                    tuple(((it.get(f) or "").casefold() for f in prim_fields)),
                    (it.get("evidence_refs")[0] if it.get("evidence_refs") else ""),
                    int(it.get("__idx", 0)),
                ),
            )
        except Exception:
            return list(lst)

    impact_items_sorted = _det_sort(impact_items, ["title", "summary", "why"])
    gaps_items_sorted = _det_sort(gaps_items, ["gap", "summary", "why"])
    actions_items_sorted = _det_sort(actions_items, ["title", "summary", "why"])

    # Remove internal index keys
    for L in (impact_items_sorted, gaps_items_sorted, actions_items_sorted):
        for it in L:
            if "__idx" in it:
                del it["__idx"]

    # Apply caps per section (max_items) and section char budgets (max_chars)
    section_meta: Dict[str, Any] = {}

    # metadata section is a dict: not capped, but keep deterministic and small.
    section_meta["metadata"] = {"truncated": False, "truncation_reasons": [], "truncation_point": None}

    # top_findings
    tf_caps = _effective_caps_for_section("top_findings")
    tf_items, tf_items_meta = _cap_items_deterministic(findings_items_sorted, tf_caps.get("max_items"))
    tf_items, tf_chars_meta = _apply_char_budget_to_items(tf_items, budget=tf_caps.get("max_chars"), fields=["title", "why"])
    # merge metas deterministically
    tf_trunc = bool(tf_items_meta.get("truncated") or tf_chars_meta.get("truncated"))
    tf_reasons = sorted(set((tf_items_meta.get("truncation_reasons") or []) + (tf_chars_meta.get("truncation_reasons") or [])))
    tf_point = tf_items_meta.get("truncation_point") if tf_items_meta.get("truncated") else tf_chars_meta.get("truncation_point")
    section_meta["top_findings"] = {"truncated": tf_trunc, "truncation_reasons": tf_reasons, "truncation_point": tf_point}

    # evidence
    ev_caps = _effective_caps_for_section("evidence")
    ev_items, ev_items_meta = _cap_items_deterministic(evidence_items_sorted, ev_caps.get("max_items"))
    ev_items, ev_chars_meta = _apply_char_budget_to_items(ev_items, budget=ev_caps.get("max_chars"), fields=["why", "excerpt"])
    ev_trunc = bool(ev_items_meta.get("truncated") or ev_chars_meta.get("truncated"))
    ev_reasons = sorted(set((ev_items_meta.get("truncation_reasons") or []) + (ev_chars_meta.get("truncation_reasons") or [])))
    ev_point = ev_items_meta.get("truncation_point") if ev_items_meta.get("truncated") else ev_chars_meta.get("truncation_point")
    section_meta["evidence"] = {"truncated": ev_trunc, "truncation_reasons": ev_reasons, "truncation_point": ev_point}

    # impact_surface
    is_caps = _effective_caps_for_section("impact_surface")
    is_items, is_items_meta = _cap_items_deterministic(impact_items_sorted, is_caps.get("max_items"))
    is_items, is_chars_meta = _apply_char_budget_to_items(is_items, budget=is_caps.get("max_chars"), fields=["why", "summary", "title"])
    is_trunc = bool(is_items_meta.get("truncated") or is_chars_meta.get("truncated"))
    is_reasons = sorted(set((is_items_meta.get("truncation_reasons") or []) + (is_chars_meta.get("truncation_reasons") or [])))
    is_point = is_items_meta.get("truncation_point") if is_items_meta.get("truncated") else is_chars_meta.get("truncation_point")
    section_meta["impact_surface"] = {"truncated": is_trunc, "truncation_reasons": is_reasons, "truncation_point": is_point}

    # gaps
    gaps_caps = _effective_caps_for_section("gaps")
    g_items, g_items_meta = _cap_items_deterministic(gaps_items_sorted, gaps_caps.get("max_items"))
    g_items, g_chars_meta = _apply_char_budget_to_items(g_items, budget=gaps_caps.get("max_chars"), fields=["gap", "why", "next_query", "summary"])
    g_trunc = bool(g_items_meta.get("truncated") or g_chars_meta.get("truncated"))
    g_reasons = sorted(set((g_items_meta.get("truncation_reasons") or []) + (g_chars_meta.get("truncation_reasons") or [])))
    g_point = g_items_meta.get("truncation_point") if g_items_meta.get("truncated") else g_chars_meta.get("truncation_point")
    section_meta["gaps"] = {"truncated": g_trunc, "truncation_reasons": g_reasons, "truncation_point": g_point}

    # suggested_actions
    sa_caps = _effective_caps_for_section("suggested_actions")
    sa_items, sa_items_meta = _cap_items_deterministic(actions_items_sorted, sa_caps.get("max_items"))
    sa_items, sa_chars_meta = _apply_char_budget_to_items(sa_items, budget=sa_caps.get("max_chars"), fields=["title", "summary", "why"])
    sa_trunc = bool(sa_items_meta.get("truncated") or sa_chars_meta.get("truncated"))
    sa_reasons = sorted(set((sa_items_meta.get("truncation_reasons") or []) + (sa_chars_meta.get("truncation_reasons") or [])))
    sa_point = sa_items_meta.get("truncation_point") if sa_items_meta.get("truncated") else sa_chars_meta.get("truncation_point")
    section_meta["suggested_actions"] = {"truncated": sa_trunc, "truncation_reasons": sa_reasons, "truncation_point": sa_point}

    # Determine top-level truncation indicator
    any_truncated = any(bool(m.get("truncated")) for m in section_meta.values() if isinstance(m, dict))

    caps_used: Dict[str, Any] = {}
    # Emit a stable, machine-readable caps_used shape.
    for sec in ("top_findings", "evidence", "impact_surface", "gaps", "suggested_actions"):
        ec = _effective_caps_for_section(sec)
        caps_used[sec] = {"max_items": ec.get("max_items"), "max_chars": ec.get("max_chars")}
    caps_used["per_item_char_limit"] = int(per_item_char_limit) if _is_number(per_item_char_limit) else None

    sections: Dict[str, Any] = {
        "metadata": metadata_section,
        "top_findings": tf_items,
        "evidence": ev_items,
        "impact_surface": is_items,
        "gaps": g_items,
        "suggested_actions": sa_items,
    }

    # Build a machine-readable truncation note per acceptance criteria 2
    truncation_note = {
        "truncated": bool(any_truncated),
        "sections_truncated": sorted([k for k, v in section_meta.items() if isinstance(v, dict) and v.get("truncated")]),
        "reasons": {k: list(v.get("truncation_reasons") or []) for k, v in section_meta.items() if isinstance(v, dict)},
    }

    # Ensure deterministic top-level ordering by returning a dict with fixed keys.
    # (Callers should still dump with sort_keys=True for byte-identical output.)
    bundle: Dict[str, Any] = {
        "sections": {k: sections.get(k) for k in section_order},
        "section_meta": {k: section_meta.get(k) for k in section_order},
        "meta": {
            "caps_used": caps_used,
            "truncated": bool(any_truncated),
            "truncation_note": truncation_note,
        },
    }
    return bundle


def deterministic_repo_scan(project_root: Path, max_files: Optional[int] = None, max_total_bytes: Optional[int] = None, max_file_bytes: Optional[int] = None, ignore_hidden: bool = True) -> List[str]:
    """Deterministically list repository files under project_root.

    - Returns a sorted list of repository-relative POSIX file paths.
    - By default ignores hidden files (names starting with '.') and common VCS
      directories like .git, .hg, .svn.
    - max_files can be used to limit the number of results (deterministic by
      sorting the final list).
    """
    root = Path(project_root)
    if not root.exists() or not root.is_dir():
        return []

    vcs_dirs = {".git", ".hg", ".svn"}
    files: List[str] = []

    # Use glob-based iteration for deterministic ordering and simpler hidden/VCS logic
    all_paths = sorted(root.rglob("*"))
    root_resolved = root.resolve()
    for p in all_paths:
        try:
            if not p.is_file():
                continue
            # Resolve path and ensure it remains within root
            try:
                resolved = p.resolve()
            except Exception:
                # If we can't resolve, skip
                continue
            try:
                _ensure_path_within_root(root_resolved, resolved)
            except Exception:
                continue
            # Build repo-relative parts
            rel_path = resolved.relative_to(root_resolved)
            parts = list(rel_path.parts)
            # Skip vcs dirs and hidden parts when requested
            if ignore_hidden:
                if any(part.startswith(".") for part in parts):
                    continue
            if any(part in vcs_dirs for part in parts):
                continue
            files.append(rel_path.as_posix())
        except Exception:
            # Skip problematic entries
            continue

    files = sorted(files)
    if isinstance(max_files, int) and max_files >= 0:
        return files[:max_files]
    return files


def extract_symbols_from_text(text: str) -> List[Dict[str, Any]]:
    """Lightweight, regex-based symbol extraction from text.

    Returns a list of dicts: {"name": <symbol_name>, "line": <1-based line>, "context": <line_text>}.
    The heuristics are intentionally lightweight and best-effort: they look for
    common definition patterns for Python and JavaScript/TypeScript and avoid
    heavy parsing. Deduplication keeps first-seen occurrence for a given name.
    """
    if not isinstance(text, str) or not text:
        return []
    symbols: List[Dict[str, Any]] = []
    seen = set()
    patterns = [
        re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
        re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
        re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
        re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*="),
        re.compile(r"^\s*export\s+(?:const|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
    ]
    for i, line in enumerate(text.splitlines()):
        line_no_newline = line.rstrip("\n\r")
        for pat in patterns:
            m = pat.match(line_no_newline)
            if m:
                name = m.group(1)
                if name in seen:
                    break
                seen.add(name)
                symbols.append({"name": name, "line": i + 1, "context": line_no_newline.strip()})
                break
    return symbols


def index_symbols(text: str) -> List[str]:
    """Return symbol names (strings) discovered in text using extract_symbols_from_text.

    Deterministic and lightweight wrapper.
    """
    syms = extract_symbols_from_text(text)
    return [s["name"] for s in syms]


def find_symbol_references_in_text(text: str, symbol_names: List[str], per_symbol_limit: int = 3) -> List[Dict[str, Any]]:
    """Find word-boundary references to the given symbol_names inside text.

    Returns a list of dicts: {"name": str, "line": int, "context": str, "count": int}
    The search is deterministic: symbol_names are iterated in sorted order and
    matches within the text are discovered by scanning lines in order. For each
    symbol the top-N (per_symbol_limit) locations are returned, chosen by
    descending count (per-line) and ascending line number as a tie-breaker.
    """
    if not isinstance(text, str) or not text or not symbol_names:
        return []
    results: List[Dict[str, Any]] = []
    # Use deterministic ordering of symbol names
    for name in sorted(set(symbol_names)):
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        occs: List[Dict[str, Any]] = []
        for i, line in enumerate(text.splitlines()):
            count = len(pat.findall(line))
            if count > 0:
                occs.append({"name": name, "line": i + 1, "context": line.strip(), "count": count})
        if not occs:
            continue
        # Sort per-line occurrences by count desc then line asc
        occs.sort(key=lambda o: (-o["count"], o["line"]))
        # Take top per_symbol_limit
        for o in occs[: max(1, int(per_symbol_limit))]:
            results.append(o)
    # Final deterministic ordering across symbols: by (name asc, line asc)
    results.sort(key=lambda r: (r["name"], r["line"]))
    return results


def extract_imports_from_text(text: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract import statements from a piece of text.

    Returns a list of dicts with keys:
      - module: the import target string as it appears (e.g. 'os', 'pkg.mod', './lib')
      - names: list of imported names or [] when not applicable
      - line: 1-based line number
      - lang: 'python' or 'js' when heuristic matched

    The heuristics are intentionally lightweight and best-effort.
    """
    if not isinstance(text, str) or not text:
        return []
    imports: List[Dict[str, Any]] = []
    py_from = re.compile(r"^\s*from\s+([A-Za-z0-9_\.\-]+|\.+[A-Za-z0-9_\.]*)\s+import\s+(.+)")
    py_import = re.compile(r"^\s*import\s+(.+)")

    # JS/TS: import ... from 'x'  OR  import 'x'  OR require('x')
    js_import_from = re.compile(r"^\s*import\s+(?:[^'\"]+\s+from\s+)?[\'\"]([^\'\"]+)[\'\"]")
    js_import_side = re.compile(r"^\s*import\s*[\'\"]([^\'\"]+)[\'\"]")
    js_require = re.compile(r"require\(\s*[\'\"]([^\'\"]+)[\'\"]\s*\)")

    for i, line in enumerate(text.splitlines()):
        # Python 'from X import Y'
        m = py_from.match(line)
        if m:
            module = m.group(1).strip()
            names_raw = m.group(2).strip()
            # strip parentheses and split
            names_clean = names_raw.strip('() ')
            names = [n.split()[0] for n in re.split(r"\s*,\s*", names_clean) if n]
            imports.append({"module": module, "names": names, "line": i + 1, "lang": "python"})
            continue
        # Python 'import a, b as c'
        m2 = py_import.match(line)
        if m2:
            names_raw = m2.group(1).strip()
            parts = [p.strip() for p in re.split(r"\s*,\s*", names_raw) if p.strip()]
            modules = []
            for p in parts:
                # 'a as b' -> 'a', 'a.b' stays as-is
                mod = p.split()[0]
                modules.append(mod)
            for mod in modules:
                imports.append({"module": mod, "names": [], "line": i + 1, "lang": "python"})
            continue
        # JS/TS import from
        m3 = js_import_from.match(line)
        if m3:
            module = m3.group(1).strip()
            imports.append({"module": module, "names": [], "line": i + 1, "lang": "js"})
            continue
        # JS/TS side-effect import
        m4 = js_import_side.match(line)
        if m4:
            module = m4.group(1).strip()
            imports.append({"module": module, "names": [], "line": i + 1, "lang": "js"})
            continue
        # require() calls (commonjs)
        for m5 in js_require.finditer(line):
            module = m5.group(1).strip()
            imports.append({"module": module, "names": [], "line": i + 1, "lang": "js"})
    return imports


def _resolve_python_relative_import(src_path: Path, module_spec: str, project_root: Path) -> Optional[str]:
    """Resolve simple python relative import spec ('..mod.sub') to a repo-relative path.

    Best-effort: counts leading dots, moves up accordingly from source directory and
    appends module parts as path. Tries .py file then package/__init__.py.
    Returns repo-relative POSIX path string when resolvable and within project_root,
    otherwise None.
    """
    if not module_spec or not module_spec.startswith('.'):
        return None
    # count leading dots
    m = re.match(r"^(\.+)(.*)$", module_spec)
    if not m:
        return None
    dots, rest = m.group(1), m.group(2)
    up = len(dots)
    base = src_path.parent
    for _ in range(up - 1):
        base = base.parent
    # rest may be '' or '.module.sub'
    rest = rest.lstrip('.')
    parts = [p for p in rest.split('.') if p]
    candidate = base
    for p in parts:
        candidate = candidate.joinpath(p)
    # try file.py
    py_candidate = candidate.with_suffix('.py')
    if py_candidate.exists():
        try:
            _ensure_path_within_root(project_root, py_candidate)
            return str(py_candidate.relative_to(project_root.resolve()).as_posix())
        except Exception:
            return None
    # try package __init__.py
    pkg_init = candidate.joinpath('__init__.py')
    if pkg_init.exists():
        try:
            _ensure_path_within_root(project_root, pkg_init)
            return str(pkg_init.relative_to(project_root.resolve()).as_posix())
        except Exception:
            return None
    return None


def _resolve_js_relative_import(src_path: Path, module_spec: str, project_root: Path, exts: List[str]) -> Optional[str]:
    """Resolve JS/TS relative imports like './mod', '../lib/util' to repo-relative path.

    Tries adding common extensions in deterministic order and index files.
    """
    if not module_spec or not (module_spec.startswith('.') or module_spec.startswith('/')):
        return None
    candidate = src_path.parent.joinpath(module_spec)
    # Normalize candidate (it may be path-like string)
    try:
        candidate_resolved = candidate.resolve()
    except Exception:
        # fallback to joining without resolving
        candidate_resolved = candidate
    # If candidate exists as-is (module_spec may already include extension), accept it
    if Path(candidate_resolved).exists():
        try:
            _ensure_path_within_root(project_root, Path(candidate_resolved))
            return str(Path(candidate_resolved).relative_to(project_root.resolve()).as_posix())
        except Exception:
            return None
    # Try direct file with extensions appended
    for ext in exts:
        f = Path(str(candidate_resolved) + ext)
        if f.exists():
            try:
                _ensure_path_within_root(project_root, f)
                return str(f.relative_to(project_root.resolve()).as_posix())
            except Exception:
                return None
    # Try index files inside directory
    for ext in exts:
        idx = candidate_resolved.joinpath('index' + ext)
        if idx.exists():
            try:
                _ensure_path_within_root(project_root, idx)
                return str(idx.relative_to(project_root.resolve()).as_posix())
            except Exception:
                return None
    return None


def build_dependency_graph(project_root: Path, *, extensions: Optional[List[str]] = None, index_all: bool = False) -> Dict[str, Dict[str, List[str]]]:
    """Build a deterministic forward and reverse dependency graph for the repo.

    Returns a dict with keys 'forward' and 'reverse', each mapping node -> sorted list
    of targets/sources. Nodes are repo-relative POSIX file paths (strings). When an
    import cannot be resolved to a repo file the original module specifier string is
    used as the target.

    - extensions: list of extensions to consider for resolution (defaults to
      typical python/js/ts extensions).
    - index_all: when True, includes files with no recognized imports as keys with
      empty lists; otherwise only files with imports are present in forward graph.
    """
    if extensions is None:
        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx']
    root = Path(project_root)
    if not root.exists() or not root.is_dir():
        return {"forward": {}, "reverse": {}}

    # Collect candidate files
    files = deterministic_repo_scan(root)
    # Filter by extensions to scan
    exts = set(extensions)
    # deterministic ordering
    files = [f for f in files if Path(f).suffix in exts]
    files = sorted(files)

    forward: Dict[str, List[str]] = {}
    reverse: Dict[str, List[str]] = {}

    for rel in files:
        src = root.joinpath(rel)
        try:
            # validate path is within root; no need to keep returned value
            _ensure_path_within_root(root.resolve(), src)
        except Exception:
            continue
        try:
            text = src.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        imports = extract_imports_from_text(text, filename=rel)
        targets: List[str] = []
        for imp in imports:
            mod = imp.get('module')
            lang = imp.get('lang')
            resolved_target: Optional[str] = None
            # Attempt resolution for relative imports
            if lang == 'python' and isinstance(mod, str) and mod.startswith('.'):
                resolved_target = _resolve_python_relative_import(src, mod, root)
            elif lang == 'js' and isinstance(mod, str) and (mod.startswith('.') or mod.startswith('/')):
                resolved_target = _resolve_js_relative_import(src, mod, root, extensions=['.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs'])
            # If resolved_target is None, keep the module specifier as target
            target = resolved_target if resolved_target is not None else mod
            if target is None:
                continue
            targets.append(target)
        # Deduplicate while preserving deterministic order
        seen = set()
        deduped = []
        for t in targets:
            if t in seen:
                continue
            seen.add(t)
            deduped.append(t)
        if deduped or index_all:
            forward[rel] = sorted(deduped)
            for t in deduped:
                reverse.setdefault(t, []).append(rel)

    # Ensure deterministic lists in reverse graph
    for k in list(reverse.keys()):
        reverse[k] = sorted(reverse[k])
    # Also sort forward keys and values deterministically when returning
    forward_sorted = {k: forward[k] for k in sorted(forward.keys())}
    return {"forward": forward_sorted, "reverse": {k: reverse[k] for k in sorted(reverse.keys())}}


def impact_surface_for_paths(project_root: Path, paths: List[str], depth: int = 2, weight_decay: float = 0.5, extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Compute a simple, scored impact surface (dep_edge evidence) for given paths.

    This best-effort helper uses the repo dependency graph (build_dependency_graph)
    and walks the reverse dependency graph starting from each provided path to
    discover files that depend on them. The traversal is breadth-first up to
    `depth` hops. Scores are assigned deterministically using an exponential
    decay: score = weight_decay ** (distance-1), where direct dependents have
    distance==1 and score==1.0 if weight_decay==1.0.

    Returned list items are dicts with keys:
      - kind: 'dep_edge'
      - target: the original file path (one of the provided inputs)
      - source: the dependent file path discovered in the repo
      - distance: integer hop count (1 for direct dependents)
      - score: float in (0,1]

    Deterministic ordering: results are sorted by (target asc, source asc, distance asc).
    """
    if extensions is None:
        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx']
    if not isinstance(paths, (list, tuple)):
        return []
    # Build graph with index_all=True so reverse entries are present for files
    graph = build_dependency_graph(project_root, extensions=extensions, index_all=True)
    reverse = graph.get('reverse', {}) or {}

    results: List[Dict[str, Any]] = []

    # Iterate inputs in deterministic order
    for target in sorted(set(paths)):
        # BFS over reverse graph up to depth
        visited: Dict[str, int] = {target: 0}
        q = deque([(target, 0)])
        while q:
            node, dist = q.popleft()
            if dist >= depth:
                continue
            # get dependents of node
            dependents = reverse.get(node, [])
            for src in sorted(dependents):
                nd = dist + 1
                # If we've already seen src with an equal or shorter distance, skip
                if src in visited and visited[src] <= nd:
                    continue
                visited[src] = nd
                # record an evidence entry for this edge (src depends on target)
                if src != target:
                    # distance is at least 1 here
                    score = float(weight_decay ** (nd - 1)) if nd >= 1 else 1.0
                    results.append({
                        "kind": "dep_edge",
                        "target": target,
                        "source": src,
                        "distance": nd,
                        "score": score,
                        "why": f"{src} depends on {target} (distance={nd})",
                    })
                q.append((src, nd))

    # Deterministic ordering
    results.sort(key=lambda r: (r.get("target", ""), r.get("source", ""), int(r.get("distance", 0))))
    return results


def extract_evidence_items(text_or_project_root: Any, *, per_symbol_limit: int = 3, overall_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Thin wrapper that produces evidence-like dicts from either text or a Path.

    - If passed a string, it is treated as file text and symbol defs/refs are
      extracted from the text excerpt only.
    - If passed a Path, gather_basic_evidence is used to scan the repository and
      the returned evidence items are propagated.

    This function is best-effort and deterministic; it is intended for tests
    and lightweight indexing use-cases.
    """
    if isinstance(text_or_project_root, str):
        text = text_or_project_root
        defs = extract_symbols_from_text(text)
        names = [d["name"] for d in defs]
        refs = find_symbol_references_in_text(text, names, per_symbol_limit=per_symbol_limit)
        evs: List[Dict[str, Any]] = []
        # Emit defs first in deterministic order
        for d in defs:
            evs.append({
                "path": "<text>",
                "kind": "symbol_def",
                "line": d["line"],
                "excerpt": d.get("context"),
                "symbol": d["name"],
                "score": 2.0,
                "symbols": [d["name"]],
                "ev_id": _canonical_ev_id(f"symbol_def:{d['name']}:{d['line']}", "<text>", "symbol_def"),
            })
        # Then refs
        for r in refs:
            evs.append({
                "path": "<text>",
                "kind": "symbol_ref",
                "line": r["line"],
                "excerpt": r.get("context"),
                "symbol": r["name"],
                "score": float(r.get("count", 1)),
                "symbols": [r["name"]],
                "ev_id": _canonical_ev_id(f"symbol_ref:{r['name']}:{r['line']}", "<text>", "symbol_ref"),
            })
        # Apply overall_limit if requested
        if isinstance(overall_limit, int) and overall_limit >= 0:
            return evs[:overall_limit]
        return evs
    elif isinstance(text_or_project_root, (Path, str)):
        # Delegate to gather_basic_evidence when provided a path
        try:
            root = Path(text_or_project_root)
            return gather_basic_evidence(root, compute_hash=False, max_items=overall_limit, per_symbol_limit=per_symbol_limit)
        except Exception:
            return []
    else:
        return []


def gather_basic_evidence(project_root: Path, *args, compute_hash: bool = False, max_items: Optional[int] = None, max_file_bytes: Optional[int] = None, max_total_bytes: Optional[int] = None, max_bytes_per_file: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
    """Best-effort, deterministic evidence gathering.

    Returns a list of dicts with at least the 'path' key (repo-relative POSIX).
    Optional keys include 'is_binary', 'size', 'sha256' and 'sha256_truncated' when
    compute_hash is requested. Hashing is performed over up to max_bytes_per_file
    bytes to bound work; when compute_hash is False the 'sha256' field is None and
    'sha256_truncated' is False.

    The function is deliberately forgiving and deterministic; it should not
    raise for unreadable files and will skip files outside project_root.

    Accepts a flexible signature so callers may pass a ResearchPlan as the
    second positional argument (ignored here) or only use keyword args as tests do.

    Additionally, this function now performs lightweight regex-based symbol
    extraction on decoded text files and includes a 'symbols' key in the returned
    evidence dict when symbols are found. This supports symbol indexing and
    symbol_ref evidence extraction downstream.
    """
    # Accept optional plan positional argument but we don't require it here.
    plan = None
    if args:
        plan = args[0]

    root = Path(project_root)
    if not root.exists() or not root.is_dir():
        return []

    # Prefer explicit per-call limits, fall back to plan.budget if available
    if max_items is None and hasattr(plan, "budget") and isinstance(plan.budget, dict):
        max_items = plan.budget.get("max_evidence_items")
    if max_file_bytes is None and hasattr(plan, "budget") and isinstance(plan.budget, dict):
        max_file_bytes = plan.budget.get("max_file_bytes")
    if max_total_bytes is None and hasattr(plan, "budget") and isinstance(plan.budget, dict):
        max_total_bytes = plan.budget.get("max_total_bytes")

    # For per-file hashing bounds, prefer explicit max_bytes_per_file; fall back
    # to max_file_bytes for compatibility.
    if max_bytes_per_file is None:
        max_bytes_per_file = max_file_bytes

    # per-symbol caps
    per_symbol_limit = int(kwargs.get("per_symbol_limit", 3))

    files = deterministic_repo_scan(root, max_files=None, max_total_bytes=max_total_bytes, max_file_bytes=max_file_bytes)

    evidence: List[Dict[str, Any]] = []
    total_read = 0
    root_resolved = root.resolve()

    # Track dedup and per-symbol counts across the repository so we can cap refs
    seen_defs = set()
    seen_refs = set()
    per_symbol_counts: Dict[str, int] = {}

    for rel in files:
        try:
            full = root.joinpath(rel)
            # Defensive: ensure within root
            try:
                resolved = _ensure_path_within_root(root_resolved, full)
            except Exception:
                # If path is invalid/outside root, skip this evidence item
                continue
            try:
                stat = resolved.stat()
            except Exception:
                continue
            size = stat.st_size

            # Decide how many bytes to read for hashing/excerpt
            if isinstance(max_bytes_per_file, int) and max_bytes_per_file >= 0:
                to_read = min(size, max_bytes_per_file)
            elif isinstance(max_file_bytes, int) and max_file_bytes >= 0:
                to_read = min(size, max_file_bytes)
            else:
                # Default: read up to 4096 bytes for excerpt/hash when requested
                to_read = min(size, 4096)

            # Bound total bytes read when requested
            if isinstance(max_total_bytes, int) and max_total_bytes >= 0:
                if total_read >= max_total_bytes:
                    break
                to_read = min(to_read, max_total_bytes - total_read)

            # Read the necessary bytes defensively
            try:
                with resolved.open("rb") as fh:
                    data = fh.read(to_read or 0)
            except Exception:
                # If unreadable, skip this file
                continue

            total_read += len(data)

            # Heuristic: if a NUL byte is present it's binary
            is_binary = False
            if b"\x00" in data:
                is_binary = True
            else:
                # Try to decode as utf-8; if it fails, treat as binary
                try:
                    _ = data.decode("utf-8")
                except Exception:
                    is_binary = True

            sha = None
            sha256_truncated = False
            if compute_hash:
                h = hashlib.sha256()
                h.update(data)
                sha = h.hexdigest()
                # If the amount hashed is less than file size, mark truncated
                if to_read < size:
                    sha256_truncated = True
            else:
                sha = None
                sha256_truncated = False

            excerpt = None
            symbols: List[Dict[str, Any]] = []
            if not is_binary:
                try:
                    excerpt = data.decode("utf-8", errors="replace")
                    if len(excerpt) > 1024:
                        excerpt = excerpt[:1024]
                    # Run lightweight symbol extraction on the decoded excerpt.
                    # This is intentionally limited to the excerpt to bound cost.
                    symbols = extract_symbols_from_text(excerpt)
                except Exception:
                    excerpt = None
                    symbols = []

            base_ev = {
                "path": rel,
                "is_binary": is_binary,
                "size": size,
                "sha256": sha,
                "sha256_truncated": sha256_truncated,
                "excerpt": excerpt,
            }

            # If symbols were found, include symbol_def evidence entries and symbol_ref entries
            if symbols:
                # Emit symbol definition evidence deterministically (first-seen preserved)
                for s in symbols:
                    name = s.get("name")
                    line = int(s.get("line", 0))
                    context = s.get("context")
                    key = ("def", name, rel, line)
                    if key in seen_defs:
                        continue
                    seen_defs.add(key)
                    ev_def = dict(base_ev)
                    ev_def.update({
                        "kind": "symbol_def",
                        "line": line,
                        "excerpt": context,
                        "symbol": name,
                        "score": 2.0,
                        "symbols": [name],
                    })
                    ev_def["ev_id"] = _canonical_ev_id(f"symbol_def:{name}:{rel}:{line}", rel, "symbol_def")
                    evidence.append(ev_def)

                # Find references within the excerpt (bounded) for the discovered symbols
                names = [s.get("name") for s in symbols if s.get("name")]
                refs = find_symbol_references_in_text(excerpt or "", names, per_symbol_limit=per_symbol_limit)
                # refs are sorted deterministically by (name, line)
                for r in refs:
                    name = r.get("name")
                    line = int(r.get("line", 0))
                    context = r.get("context")
                    count = int(r.get("count", 1))
                    # Enforce per-symbol global cap
                    cur = per_symbol_counts.get(name, 0)
                    if cur >= per_symbol_limit:
                        continue
                    per_symbol_counts[name] = cur + 1
                    key = ("ref", name, rel, line)
                    if key in seen_refs:
                        continue
                    seen_refs.add(key)
                    ev_ref = dict(base_ev)
                    ev_ref.update({
                        "kind": "symbol_ref",
                        "line": line,
                        "excerpt": context,
                        "symbol": name,
                        "score": float(count),
                        "symbols": [name],
                    })
                    ev_ref["ev_id"] = _canonical_ev_id(f"symbol_ref:{name}:{rel}:{line}", rel, "symbol_ref")
                    evidence.append(ev_ref)
            else:
                # No symbols found; include the basic file evidence as before
                evidence.append(base_ev)
        except Exception:
            # Skip this file on any unexpected error and continue
            continue

    # Deterministic global sorting and capping before returning
    def _score_for_sort(ev: Dict[str, Any]) -> float:
        try:
            return float(ev.get("score", 0) or 0)
        except Exception:
            return 0.0

    # Sort by score desc, path asc, line asc, symbol asc to deterministically break ties
    evidence.sort(key=lambda ev: (
        -_score_for_sort(ev),
        ev.get("path", ""),
        int(ev.get("line", 0)) if ev.get("line") is not None else 0,
        ev.get("symbol", ""),
    ))

    if isinstance(max_items, int) and max_items >= 0:
        return evidence[:max_items]
    return evidence


class DeepResearchEngine:
    """A minimal, safe orchestration entrypoint for deep repository research.

    Responsibilities implemented here (skeleton):
    - Validate project_root is a directory and keep it root-locked.
    - Provide deterministic query normalization and cache-key computation.
    - Best-effort repo_version lookup (git) inside run(); fall back to 'unknown'.
    - Return a schema-shaped ResearchBrief placeholder with meta.errors == ['not_implemented'].

    No filesystem or subprocess actions occur at import time.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        scope: str = "repo",
        scope_paths: Optional[List[str]] = None,
        depth: str = "standard",
        force_refresh: bool = False,
        budgets: Optional[Dict[str, Any]] = None,
        struct: Optional[Any] = None,
        llm: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Keep previous semantics but make constructor accept the wrapper-friendly
        # arguments required by the API wrapper and tests.
        self.project_root = Path(project_root)
        self.struct = struct
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        # New attributes exposed for wrapper-friendly construction
        self.scope = scope
        self.scope_paths = list(scope_paths or [])
        self.depth = depth
        self.force_refresh = bool(force_refresh)
        self.budgets = dict(budgets) if isinstance(budgets, dict) else (budgets if budgets is not None else None)
        # Backing storage for a cached repo version; tests can set/reset this via
        # set_repo_version / get_repo_version to make cache-key behavior deterministic
        # across invocations.
        self.repo_version: Optional[str] = None

    def _normalize_query(self, query: str) -> str:
        """Deterministically normalize a query string.

        Determinism rules:
        - Normalize Unicode to NFKC to avoid equivalent-but-different codepoints.
        - Collapse all internal whitespace sequences to a single space.
        - Use casefold() for deterministic case normalization across locales.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        # Normalize Unicode form, collapse whitespace, and perform casefold for
        # more aggressive and deterministic case normalization than lower().
        s = unicodedata.normalize("NFKC", query)
        collapsed = " ".join(s.split())
        normalized = collapsed.casefold()
        return normalized

    # Public alias so callers/tests can rely on normalize_query being available
    def normalize_query(self, query: str) -> str:
        return self._normalize_query(query)

    def _compute_cache_key(self, normalized_query: str, request: ResearchRequest, repo_version: str) -> str:
        """Compute a deterministic cache key for the preflight inputs.

        Uses stable JSON serialization (sort_keys=True) and sha256.
        The payload includes normalized_query, scope, sorted scope_paths, depth,
        force_refresh, budgets (as-is), and repo_version.

        When deep_research_cache.compute_cache_key is available, delegate to it
        and explicitly pass resolved_profile and resolved_budget to ensure the
        cache key source-of-truth is centralized and deterministic.
        """
        # If the external cache helper is available, prefer it as the source of truth
        try:
            if deep_research_cache is not None and hasattr(deep_research_cache, "compute_cache_key"):
                # Build the normalized_request payload expected by the helper
                normalized_request = {
                    "normalized_query": normalized_query,
                    "scope": request.scope,
                    "scope_paths": sorted(request.scope_paths or []),
                    "depth": request.depth,
                    "force_refresh": bool(request.force_refresh),
                }
                # Resolve profile and budget deterministically and pass them through
                try:
                    resolved_profile = request.depth if isinstance(request.depth, str) else None
                    resolved_budget = resolve_budget(request.depth if isinstance(request.depth, str) else None, request.budgets if isinstance(request.budgets, dict) else None)
                except Exception:
                    resolved_profile = request.depth if isinstance(request.depth, str) else None
                    resolved_budget = request.budgets if isinstance(request.budgets, dict) else {}
                # Call helper with explicit params; allow it to raise/fallback
                try:
                    return deep_research_cache.compute_cache_key(normalized_request, repo_version, resolved_profile=resolved_profile, resolved_budget=resolved_budget)
                except TypeError:
                    # Older helpers might not accept the keyword-only args; fall back to passing via payload
                    # Embed resolved values into normalized_request to maintain determinism
                    normalized_request = dict(normalized_request)
                    normalized_request["resolved_profile"] = resolved_profile
                    normalized_request["resolved_budget"] = resolved_budget
                    return deep_research_cache.compute_cache_key(normalized_request, repo_version)
                except Exception:
                    # Fall back to local computation below
                    pass
        except Exception:
            # Any unexpected error talking to external helper -> fall back
            pass

        # Fallback local deterministic payload
        try:
            resolved_budget = resolve_budget(request.depth if isinstance(request.depth, str) else None, request.budgets if isinstance(request.budgets, dict) else None)
        except Exception:
            resolved_budget = request.budgets or {}

        payload = {
            "normalized_query": normalized_query,
            "scope": request.scope,
            "scope_paths": sorted(request.scope_paths or []),
            "depth": request.depth,
            "force_refresh": bool(request.force_refresh),
            # Use the resolved budget mapping as part of the cache key for determinism
            "resolved_budget": resolved_budget,
            "repo_version": repo_version,
        }
        # Stable serialization
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return digest

    def compute_cache_key(self, query: str, request: Optional[ResearchRequest] = None, repo_version: Optional[str] = None) -> str:
        """Public helper to compute a cache key from a query string.

        If request is None a minimal ResearchRequest(query=query) is created.
        If repo_version is None the engine's get_repo_version() is used (which may
        be backed by a cached self.repo_version set via set_repo_version).
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized_query = self.normalize_query(query)
        if request is None:
            request = ResearchRequest(query=query)
        if repo_version is None:
            repo_version = self.get_repo_version()
        return self._compute_cache_key(normalized_query, request, repo_version)

    def _ensure_within_project_root(self, path: Path) -> Path:
        """Resolve a candidate path and assert it remains under project_root.

        This helper enforces the root-locked policy for any future path resolution.
        It does not perform any I/O other than path resolution.
        """
        resolved_root = self.project_root.resolve()
        resolved = Path(path).resolve()
        # Use os.path.commonpath to avoid naive prefix checks
        common = os.path.commonpath([str(resolved_root), str(resolved)])
        if common != str(resolved_root):
            raise ValueError(f"path {resolved} is outside of project root {resolved_root}")
        return resolved

    def _get_repo_version(self) -> str:
        """Best-effort attempt to read git HEAD sha within project_root.

        This is executed at run-time (not import-time) and suppresses errors,
        returning 'unknown' if git is not available or the command fails.
        """
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            # Any exception -> unknown; keep this best-effort and safe
            pass
        return "unknown"

    def get_repo_version(self) -> str:
        """Return a stored repo version if present; otherwise attempt to read it.

        This method is backed by the self.repo_version attribute so tests may set
        a stable value via set_repo_version() to make cache-key computations
        deterministic.
        """
        if self.repo_version is not None:
            return self.repo_version
        v = self._get_repo_version()
        # Cache the discovered value so future calls are stable unless tests call
        # set_repo_version to override.
        self.repo_version = v
        return v

    def set_repo_version(self, v: Optional[str]) -> None:
        """Set (or clear) the stored repo_version value used by get_repo_version().

        Passing None will clear any cached/stored value so get_repo_version() will
        re-probe the repository on next call.
        """
        if v is None:
            self.repo_version = None
        else:
            if not isinstance(v, str):
                raise TypeError("repo_version must be a string or None")
            self.repo_version = v

    def _merge_budgets_deterministic(self, plan_budget: Dict[str, Any], cached_budget: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        """Instance wrapper around module-level deterministic merge.

        Kept for backward-compatibility internally; delegates to the module
        level implementation to ensure consistent behavior.
        """
        return _merge_budgets_deterministic(plan_budget, cached_budget, keys)

    def _normalize_cached_payload(self, cached: Any) -> Optional[Dict[str, Any]]:
        """Normalize a cached json value to a plain dict payload.

        Supports read_json_cache returning either:
        - a plain dict
        - a wrapper {"__meta__": {...}, "value": <dict>}

        Returns None if the payload is not a dict.
        """
        if not isinstance(cached, dict):
            return None
        if "value" in cached and isinstance(cached.get("value"), dict):
            return cached.get("value")
        return cached

    def run_phase0(self, request, project_root: Optional[Path] = None, repo_brief_hash: Optional[str] = None, provided_run_id: Optional[str] = None) -> Any:
        """Phase 0: deterministic preflight + persisted artifact.

        This method supports two call styles (backwards-compatible):
        - run_phase0(request: ResearchRequest, provided_run_id: Optional[str]=None)
          returns a PreflightResult dataclass instance (existing behavior).
        - run_phase0(request: dict, project_root: Path|str, repo_brief_hash: str, provided_run_id: Optional[str]=None)
          is a public wrapper used by tests and callers that pass raw dicts; it
          returns a plain dict containing at minimum: schema_version, run_id,
          cache_key, cache_refs (as a dict), and normalized_request.

        The wrapper enforces root-lock by resolving project_root and only writing
        artifacts under <project_root>/.aidev/. When deep_research_cache is
        available the wrapper will call deep_research_cache.compute_cache_key to
        compute the cache key; otherwise it falls back to a stable sha256 over
        the normalized request and repo_brief_hash.
        """
        # Backwards-compatible path: ResearchRequest instance -> existing behavior
        if isinstance(request, ResearchRequest):
            # preserve previous signature semantics
            provided_run_id_local = provided_run_id

            # Root lock validation
            if not self.project_root.exists():
                raise ValueError(f"project_root {self.project_root} does not exist")
            if not self.project_root.is_dir():
                raise ValueError(f"project_root {self.project_root} is not a directory")

            normalized_query = self.normalize_query(request.query)
            normalized_request = {
                "normalized_query": normalized_query,
                "scope": request.scope,
                "scope_paths": sorted(request.scope_paths or []),
                "depth": request.depth,
                "force_refresh": bool(request.force_refresh),
            }

            repo_version = self.get_repo_version()
            cache_key = self._compute_cache_key(normalized_query, request, repo_version)
            run_id = provided_run_id_local or cache_key

            # Deterministic repository-relative cache path under .aidev/
            cache_rel = (Path(".aidev") / "preflight" / f"{run_id}.json").as_posix()

            _emit_deep_research_event(
                DEEP_RESEARCH_PHASE_STARTED,
                {
                    "phase": "phase0",
                    "run_id": run_id,
                    "cache_key": cache_key,
                    "cache_ref": cache_rel,
                    "repo_version": repo_version,
                },
                logger=self.logger,
            )

            # Initialize effective budget (include token bookkeeping fields)
            effective_budget = resolve_budget(request.depth, request.budgets)

            # Best-effort deterministic repo scan, bounded
            repo_index: Optional[List[str]] = None
            try:
                scan_kwargs = {
                    "max_files": effective_budget.get("max_files_touched", DEFAULT_BUDGET["max_files_touched"]),
                    "max_total_bytes": effective_budget.get("max_total_bytes", DEFAULT_BUDGET["max_total_bytes"]),
                    "max_file_bytes": effective_budget.get("max_file_bytes", DEFAULT_BUDGET["max_file_bytes"]),
                }
                repo_index = deterministic_repo_scan(self.project_root, **scan_kwargs)
            except Exception as e:
                if self.logger:
                    self.logger.debug("DeepResearchEngine.run_phase0: deterministic_repo_scan failed: %s", e)
                repo_index = None

            created_at = _utc_now_iso_z()

            preflight = PreflightResult(
                schema_version=1,
                run_id=run_id,
                cache_refs=[cache_rel],
                normalized_request=normalized_request,
                budget=effective_budget,
                repo_version=repo_version,
                repo_index=repo_index,
                created_at=created_at,
            )

            # Best-effort persist (atomic + root-locked)
            wrote_ok = False
            write_err: Optional[str] = None
            try:
                # Ensure rel path doesn't try to escape root via join tricks
                self._ensure_within_project_root(self.project_root.joinpath(cache_rel))
                write_json_cache(self.project_root, cache_rel, dataclasses.asdict(preflight))
                wrote_ok = True
                _emit_deep_research_event(
                    DEEP_RESEARCH_ARTIFACT_WRITTEN,
                    {
                        "phase": "phase0",
                        "run_id": run_id,
                        "artifact_type": "preflight",
                        "cache_ref": cache_rel,
                    },
                    logger=self.logger,
                )
                if self.logger:
                    self.logger.debug("DeepResearchEngine.phase0: wrote preflight cache_rel=%s run_id=%s", cache_rel, run_id)
            except Exception as e:
                write_err = _safe_err_str(e)
                if self.logger:
                    self.logger.debug("DeepResearchEngine.phase0: failed to write preflight artifact %s: %s", cache_rel, e)

            _emit_deep_research_event(
                DEEP_RESEARCH_PHASE_DONE,
                {
                    "phase": "phase0",
                    "run_id": run_id,
                    "cache_key": cache_key,
                    "cache_ref": cache_rel,
                    "status": "ok" if wrote_ok else "error",
                    "error": write_err,
                },
                logger=self.logger,
            )

            return preflight

        # New public wrapper signature: request is a dict
        if isinstance(request, dict):
            if project_root is None:
                raise TypeError("project_root must be provided when request is a dict")
            root = Path(project_root)
            if not root.exists():
                raise ValueError(f"project_root {root} does not exist")
            if not root.is_dir():
                raise ValueError(f"project_root {root} is not a directory")

            # Normalize incoming request dict into the expected normalized_request
            q = request.get("query", "")
            normalized_query = self.normalize_query(q if isinstance(q, str) else "")
            scope = request.get("scope", "repo")
            scope_paths = list(request.get("scope_paths", []) or [])
            depth = request.get("depth", "standard")
            force_refresh = bool(request.get("force_refresh", False))

            normalized_request = {
                "normalized_query": normalized_query,
                "scope": scope,
                "scope_paths": sorted(scope_paths),
                "depth": depth,
                "force_refresh": force_refresh,
            }

            # Budget initialization and token bookkeeping
            effective_budget = resolve_budget(str(depth) if isinstance(depth, str) else None, request.get("budgets") if isinstance(request.get("budgets"), dict) else None)

            # Compute cache_key using deep_research_cache when available; otherwise
            # fall back to a stable sha256 over the normalized request + repo_brief_hash.
            if deep_research_cache is not None and hasattr(deep_research_cache, "compute_cache_key"):
                try:
                    # Pass resolved_profile and resolved_budget explicitly per contract
                    resolved_profile = depth if isinstance(depth, str) else None
                    resolved_budget = effective_budget
                    cache_key = deep_research_cache.compute_cache_key(normalized_request, repo_brief_hash, resolved_profile=resolved_profile, resolved_budget=resolved_budget)
                except TypeError:
                    # Older signature may not accept keywords; embed resolved fields into request
                    nr = dict(normalized_request)
                    nr["resolved_profile"] = depth if isinstance(depth, str) else None
                    nr["resolved_budget"] = effective_budget
                    try:
                        cache_key = deep_research_cache.compute_cache_key(nr, repo_brief_hash)
                    except Exception:
                        serialized = json.dumps({"normalized_request": normalized_request, "repo_brief_hash": repo_brief_hash or ""}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                except Exception:
                    serialized = json.dumps({"normalized_request": normalized_request, "repo_brief_hash": repo_brief_hash or ""}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            else:
                serialized = json.dumps({"normalized_request": normalized_request, "repo_brief_hash": repo_brief_hash or ""}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            run_id = provided_run_id or cache_key

            # Deterministic repository-relative cache path under .aidev/
            cache_rel = (Path(".aidev") / "preflight" / f"{run_id}.json").as_posix()

            _emit_deep_research_event(
                DEEP_RESEARCH_PHASE_STARTED,
                {
                    "phase": "phase0",
                    "run_id": run_id,
                    "cache_key": cache_key,
                    "cache_ref": cache_rel,
                    "repo_version": repo_brief_hash or "unknown",
                },
                logger=self.logger,
            )

            created_at = _utc_now_iso_z()

            # Compose the structured preflight result as a plain dict (stable shape)
            preflight_dict: Dict[str, Any] = {
                "schema_version": 1,
                "run_id": run_id,
                "cache_key": cache_key,
                # cache_refs as an object/dict so callers can inspect the deterministic path
                "cache_refs": {"preflight": cache_rel},
                "normalized_request": normalized_request,
                "budget": effective_budget,
                "repo_version": repo_brief_hash or "unknown",
                "repo_index": None,
                "created_at": created_at,
            }

            # Best-effort deterministic repo scan, bounded (do not pass unsupported kwargs)
            try:
                scan_kwargs = {
                    "max_files": effective_budget.get("max_files_touched", DEFAULT_BUDGET["max_files_touched"]),
                    "max_total_bytes": effective_budget.get("max_total_bytes", DEFAULT_BUDGET["max_total_bytes"]),
                    "max_file_bytes": effective_budget.get("max_file_bytes", DEFAULT_BUDGET["max_file_bytes"]),
                }
                repo_index = deterministic_repo_scan(root, **scan_kwargs)
                preflight_dict["repo_index"] = repo_index
            except Exception as e:
                if self.logger:
                    self.logger.debug("DeepResearchEngine.run_phase0(dict): deterministic_repo_scan failed: %s", e)
                preflight_dict["repo_index"] = None

            # Persist artifact under the provided project_root/.aidev/
            wrote_ok = False
            write_err: Optional[str] = None
            try:
                # Ensure the resolved target path is within the provided root
                _ensure_path_within_root(root, root.joinpath(cache_rel))
                write_json_cache(root, cache_rel, preflight_dict)
                wrote_ok = True
                _emit_deep_research_event(
                    DEEP_RESEARCH_ARTIFACT_WRITTEN,
                    {
                        "phase": "phase0",
                        "run_id": run_id,
                        "artifact_type": "preflight",
                        "cache_ref": cache_rel,
                    },
                    logger=self.logger,
                )
                if self.logger:
                    self.logger.debug("DeepResearchEngine.phase0(dict): wrote preflight cache_rel=%s run_id=%s", cache_rel, run_id)
            except Exception as e:
                write_err = _safe_err_str(e)
                if self.logger:
                    self.logger.debug("DeepResearchEngine.phase0(dict): failed to write preflight artifact %s: %s", cache_rel, e)

            _emit_deep_research_event(
                DEEP_RESEARCH_PHASE_DONE,
                {
                    "phase": "phase0",
                    "run_id": run_id,
                    "cache_key": cache_key,
                    "cache_ref": cache_rel,
                    "status": "ok" if wrote_ok else "error",
                    "error": write_err,
                },
                logger=self.logger,
            )

            return preflight_dict

        # Unsupported request type
        raise TypeError("request must be a ResearchRequest instance or a dict when using the wrapper signature")

    def plan(self, request: ResearchRequest, repo_meta: Optional[Dict[str, Any]] = None) -> ResearchPlan:
        """Validate a ResearchRequest and produce a safe ResearchPlan.

        NOTE: Signature updated to accept repo_meta: callers can provide repository
        metadata (a deterministic, read-only dict) that could be used to compose
        prompts for an LLM planner. This implementation is intentionally
        conservative: it accepts repo_meta but currently performs the same
        validated, fail-closed planning behavior as before. Future work may use
        repo_meta when calling an external LLM.

        This method performs schema-like validation on the incoming request, normalizes
        the query deterministically, merges provided budgets with defaults (clamping
        numeric values), and returns a ResearchPlan. If validation fails the method
        returns a fail-closed ResearchPlan with meta.errors populated and conservative
        budget values (zeros) to ensure downstream execution remains safe.
        """
        validation_errors: List[str] = []
        safe_fallback = False
        try:
            if not isinstance(request, ResearchRequest):
                raise TypeError("request must be a ResearchRequest instance")

            # Query validation
            if not isinstance(request.query, str) or not request.query.strip():
                validation_errors.append("query must be a non-empty string")

            # Scope validation
            if request.scope not in ("repo", "subtree", "targets"):
                validation_errors.append(f"scope '{request.scope}' is not one of 'repo','subtree','targets'")

            # Depth validation
            if request.depth not in ("quick", "standard", "deep"):
                validation_errors.append(f"depth '{request.depth}' is not one of 'quick','standard','deep'")

            # Scope paths: if provided, ensure they resolve within project_root
            scope_paths_valid: List[str] = []
            if request.scope_paths:
                if not isinstance(request.scope_paths, (list, tuple)):
                    validation_errors.append("scope_paths must be a list of repo-relative paths")
                else:
                    for p in request.scope_paths:
                        if not isinstance(p, str) or not p:
                            validation_errors.append(f"invalid scope path: {p!r}")
                            continue
                        try:
                            candidate = self.project_root.joinpath(p)
                            resolved = self._ensure_within_project_root(candidate)
                            rel = Path(resolved).relative_to(self.project_root.resolve()).as_posix()
                            scope_paths_valid.append(rel)
                        except Exception:
                            validation_errors.append(f"scope_path '{p}' is outside of project_root or invalid")

            # Budgets: prefer caller-provided numeric values but clamp and merge with defaults
            provided_budgets = request.budgets if isinstance(request.budgets, dict) else {}
            try:
                # Resolve through named profiles + hard caps for production budget behavior.
                effective_budget = resolve_budget(request.depth, provided_budgets)
            except Exception:
                validation_errors.append("budget merge failed; using defaults")
                effective_budget = resolve_budget(request.depth, None)

            # If any validation errors exist, produce a safe, fail-closed budget
            if validation_errors:
                safe_fallback = True
                # Conservative zeroed budget to avoid performing I/O/LLM work
                zero_budget = {}
                for k, v in DEFAULT_BUDGET.items():
                    zero_budget[k] = 0 if _is_number(v) else v
                effective_budget = zero_budget

            normalized_query = self.normalize_query(request.query if isinstance(request.query, str) else "")

            plan = ResearchPlan(
                query=request.query if isinstance(request.query, str) else "",
                normalized_query=normalized_query,
                scope=request.scope,
                scope_paths=scope_paths_valid,
                depth=request.depth,
                subquestions=[],
                priority_targets=(list(request.scope_paths) if request.scope == "targets" else []),
                impact_hypotheses=[],
                budget=effective_budget,
                meta={
                    "created_at": _utc_now_iso_z(),
                    "validation_errors": list(validation_errors) if validation_errors else [],
                    "safe_fallback": bool(safe_fallback),
                },
            )
            return plan
        except Exception as e:
            # Fail-closed: on any unexpected exception, return a minimal plan with errors
            if self.logger:
                self.logger.debug("DeepResearchEngine.plan: unexpected error during planning: %s", e)
            zero_budget = {}
            for k, v in DEFAULT_BUDGET.items():
                zero_budget[k] = 0 if _is_number(v) else v
            return ResearchPlan(
                query=getattr(request, "query", "") if isinstance(request, ResearchRequest) else "",
                normalized_query=(self.normalize_query(request.query) if isinstance(request, ResearchRequest) and isinstance(request.query, str) else ""),
                scope=(request.scope if isinstance(request, ResearchRequest) and isinstance(request.scope, str) else "repo"),
                scope_paths=[],
                depth=(request.depth if isinstance(request, ResearchRequest) and isinstance(request.depth, str) else "standard"),
                subquestions=[],
                priority_targets=[],
                impact_hypotheses=[],
                budget=zero_budget,
                meta={"created_at": _utc_now_iso_z(), "validation_errors": [str(e)], "safe_fallback": True},
            )

    def run_phase1_plan(self, request: Union[ResearchRequest, Dict[str, Any]], preflight: Optional[PreflightResult] = None) -> ResearchPlan:
        """Phase 1: create/load a schema-valid ResearchPlan and return it.

        Minimal safe implementation: normalize and validate via self.plan and return
        the resulting ResearchPlan. This avoids introducing NotImplementedError
        stubs while preserving backward-compatible behavior for callers that rely
        on run_phase1_plan completing.
        """
        try:
            if isinstance(request, dict):
                # Convert dict->ResearchRequest conservatively
                req = ResearchRequest(
                    query=str(request.get("query", "") or ""),
                    scope=request.get("scope", "repo"),
                    scope_paths=list(request.get("scope_paths", []) or []),
                    depth=request.get("depth", "standard"),
                    force_refresh=bool(request.get("force_refresh", False)),
                    budgets=request.get("budgets") if isinstance(request.get("budgets"), dict) else None,
                )
            elif isinstance(request, ResearchRequest):
                req = request
            else:
                raise TypeError("request must be a ResearchRequest or dict")
            plan = self.plan(req)
            return plan
        except Exception as e:
            if self.logger:
                self.logger.debug("run_phase1_plan failed: %s", e)
            # Return a safe fallback plan
            return ResearchPlan(query=str(getattr(request, "query", "") or ""), normalized_query=(self.normalize_query(getattr(request, "query", "") or "") if isinstance(request, ResearchRequest) else ""), budget=resolve_budget(None, None), meta={"error": _safe_err_str(e)})

    def run_phase2_gather(self, plan: ResearchPlan, preflight: Optional[PreflightResult] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
        """Phase 2: gather evidence according to the plan.

        Minimal safe implementation: run a bounded gather_basic_evidence and return
        a dict containing an 'evidence' list. This keeps behavior deterministic and
        avoids introducing NotImplementedError stubs.
        """
        try:
            max_items = None
            if isinstance(plan, ResearchPlan) and isinstance(plan.budget, dict):
                max_items = int(plan.budget.get("max_evidence_items") or 0) if _is_number(plan.budget.get("max_evidence_items")) else None
            evs = gather_basic_evidence(self.project_root, plan, compute_hash=False, max_items=max_items, per_symbol_limit=int(plan.budget.get("per_symbol_limit", 3) if isinstance(plan.budget, dict) else 3))
            return {"ok": True, "evidence": evs}
        except Exception as e:
            if self.logger:
                self.logger.debug("run_phase2_gather failed: %s", e)
            return {"ok": False, "error": _safe_err_str(e), "evidence": []}

    def synthesize(self, plan: ResearchPlan, evidence: List[Any], request: ResearchRequest, meta: Dict[str, Any]) -> ResearchBrief:
        """Synthesize a ResearchBrief from plan + evidence.

        Minimal safe synthesis: assemble a ResearchBrief preserving evidence ids and
        filling the basic fields. This avoids stubbing out behavior and keeps
        downstream consumers operational.
        """
        try:
            evs: List[EvidenceItem] = []
            for e in (evidence or []):
                try:
                    if isinstance(e, EvidenceItem):
                        evs.append(e)
                    elif isinstance(e, dict):
                        ev_id = e.get("ev_id") or e.get("id") or _canonical_ev_id(None, e.get("path", ""), e.get("kind", ""))
                        path = e.get("path") or ""
                        kind = e.get("kind") or e.get("type") or "file"
                        lines = e.get("lines")
                        excerpt = e.get("excerpt")
                        why = e.get("why") or ""
                        score = float(e.get("score")) if _is_number(e.get("score")) else None
                        evs.append(EvidenceItem(ev_id=ev_id, path=path, kind=kind, lines=lines, excerpt=excerpt, why=why, score=score))
                except Exception:
                    continue

            brief = ResearchBrief(
                query=plan.query if isinstance(plan, ResearchPlan) else (request.query if isinstance(request, ResearchRequest) else ""),
                subquestions=list(plan.subquestions) if isinstance(plan, ResearchPlan) else [],
                evidence=evs,
                findings=[],
                impact_surface=[],
                suggested_actions=[],
                gaps=[],
                meta={"llm_calls_used": 0, "errors": []},
            )
            _emit_deep_research_event(DEEP_RESEARCH_SYNTH_DONE, {"run_id": getattr(meta, "run_id", None), "brief_len": len(evs)}, logger=self.logger)
            return brief
        except Exception as e:
            if self.logger:
                self.logger.debug("synthesize failed: %s", e)
            return ResearchBrief(query=(plan.query if isinstance(plan, ResearchPlan) else ""), meta={"errors": [str(e)], "llm_calls_used": 0})

    def _call_verify_llm(self, payload: Dict[str, Any], *, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal verify LLM call placeholder: return a skipped response.

        Real verification wiring should replace this; keeping a no-op response
        avoids runtime regressions and preserves existing callers' expectations.
        """
        return {"ok": False, "skipped": True, "reason": "verify_not_configured"}

    def _mini_gather_for_requests(self, additional_requests: List[Dict[str, str]], plan: ResearchPlan) -> List[Dict[str, Any]]:
        """Perform tiny additional gathers for ad-hoc requests.

        Minimal safe implementation returns an empty list. Callers can override
        or the function can be expanded later. Returning [] preserves callers
        that expect a list without causing failures.
        """
        return []

    def _merge_and_dedupe_evidence(self, existing: List[Any], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge two evidence lists deterministically and dedupe by ev_id or (path,line).

        Preserves first-seen order from `existing` and appends new unique items.
        """
        out: List[Dict[str, Any]] = []
        seen = set()
        def _key(e: Any) -> str:
            if isinstance(e, EvidenceItem):
                return e.ev_id or f"{e.path}:{e.lines}"
            if isinstance(e, dict):
                ev_id = e.get("ev_id") or e.get("id")
                if ev_id:
                    return ev_id
                return f"{e.get('path','')}:{e.get('line','') }"
            return str(e)

        for e in (existing or []):
            k = _key(e)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        for e in (new or []):
            k = _key(e)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out

    def _should_run_verify_for_standard(self, brief: ResearchBrief, plan: ResearchPlan, request: ResearchRequest) -> bool:
        """Decide whether to run verification for a standard plan.

        Minimal conservative policy: do not run verify by default to avoid extra LLM calls.
        """
        return False

    def verify_and_refine(self, plan: ResearchPlan, evidence: List[Any], brief: ResearchBrief, request: ResearchRequest) -> ResearchBrief:
        """Verify and optionally refine a brief. Minimal implementation returns brief unchanged.

        This keeps prior behavior stable and avoids introducing NotImplementedError
        stubs that would break callers.
        """
        return brief

    def _run_request(self, request: ResearchRequest) -> ResearchBrief:
        """High-level orchestration: plan -> gather -> synthesize -> verify.

        Minimal safe orchestration that uses the helper methods implemented above.
        """
        plan = self.plan(request)
        gather_res = self.run_phase2_gather(plan)
        evs = gather_res.get("evidence") if isinstance(gather_res, dict) else []
        brief = self.synthesize(plan, evs or [], request, {})
        if self._should_run_verify_for_standard(brief, plan, request):
            brief = self.verify_and_refine(plan, evs or [], brief, request)
        return brief

    def run(self, request: Union[ResearchRequest, str]) -> Any:
        # Support a string convenience form and the ResearchRequest form.
        if isinstance(request, str):
            rr = ResearchRequest(query=request)
            try:
                return self._run_request(rr)
            except Exception as e:
                return ResearchBrief(query=rr.query, meta={"errors": [str(e)]})
        if isinstance(request, ResearchRequest):
            try:
                return self._run_request(request)
            except Exception as e:
                return ResearchBrief(query=request.query, meta={"errors": [str(e)]})
        raise TypeError("request must be a ResearchRequest or string")


if __name__ == "__main__":
    # Minimal demonstration (safe at module run-time). This will attempt to run
    # git in the current directory when executed directly. It is not executed at
    # import time by tests that import this module.
    import sys

    logging.basicConfig(level=logging.DEBUG)
    root = Path(".").resolve()
    engine = DeepResearchEngine(root)
    req = ResearchRequest(query="Example: How does X work in this repo?", scope="repo")
    try:
        brief = engine.run(req)
        print("ResearchBrief meta:", json.dumps(brief.meta, indent=2))
    except Exception as e:
        print("Error running DeepResearchEngine:", e, file=sys.stderr)
