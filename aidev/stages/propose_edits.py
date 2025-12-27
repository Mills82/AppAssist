# aidev/stages/propose_edits.py
from __future__ import annotations

import time
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..io_utils import (
    _read_file_text_if_exists,
    apply_unified_patch,
    generate_unified_diff,
)

# Structured logging helper introduced by the logging standardization rec.
# Keep the stdlib logging import for third-party libs; prefer 'alog' below for
# structured orchestration logs.
from .. import logger as alog

# Avoid importing generate_edits at module import time to prevent circular
# imports when generate_edits imports this module (tests inspect.getsource on
# generate_edits which can fail if modules are partially-initialized).
# Use TYPE_CHECKING to keep types for static checks while performing a local
# import at call-time for the runtime function.
if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .generate_edits import ChatJsonFn, JsonSchema
else:
    ChatJsonFn = Any
    JsonSchema = Any

ProgressFn = Callable[[str, Dict[str, Any]], None]
ErrorFn = Callable[[str, Dict[str, Any]], None]
TimeoutRunner = Callable[[Callable[[], Any], float, str], Any]
TraceFn = Callable[[str, str, Dict[str, Any]], None]

TargetsEnvelope = Dict[str, Any]

# Logical stage label used when resolving models via config.get_model_for.
STAGE_LABEL = "propose_edits"


@dataclass
class FileEditContext:
    """
    Minimal, high-signal bundle of information for editing a single file.

    The goal is to avoid re-sending large multi-file snippets and instead
    provide:
      * the full current contents of the file being edited
      * any structured per-file analysis (plan/constraints/notes)
      * compact recommendation metadata
      * compact target-selection metadata

    New: routing_reason is an optional hint consumers can set to request
    a particular routing of the edit request (for example 'repair').
    Callers/analysis should prefer setting file_ctx.routing_reason = 'repair'
    rather than relying on loosely-related flags; generate/propose stages
    will use this hint to record intent while keeping the default stage
    parameter stable (STAGE_LABEL).
    """

    path: str
    language: str
    content: str
    analysis: Dict[str, Any]
    rec_meta: Dict[str, Any]
    target_meta: Dict[str, Any]
    routing_reason: Optional[str] = None


def _guess_language_from_path(path: str) -> str:
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith((".js", ".jsx", ".ts", ".tsx")):
        return "javascript"
    if p.endswith(".css"):
        return "css"
    if p.endswith((".html", ".htm")):
        return "html"
    if p.endswith((".md", ".markdown")):
        return "markdown"
    return "text"


def _why_line(rec_obj: Dict[str, Any]) -> str:
    """
    Build a short human-readable explanation for why a recommendation exists.
    """
    title = (rec_obj.get("title") or "").strip()
    # Prefer an explicit "reason", fall back to "why" or "summary".
    reason = (
        rec_obj.get("reason")
        or rec_obj.get("why")
        or rec_obj.get("summary")
        or ""
    )
    reason = reason.strip()

    if reason:
        reason_norm = re.sub(r"\s+", " ", reason)
        reason_norm = reason_norm[:200]
    else:
        reason_norm = ""

    return (f"{title} — {reason_norm}" if title else (reason_norm or "LLM suggested change")).strip()


def _build_rec_meta(rec: Dict[str, Any], focus: str) -> Dict[str, Any]:
    rid = str(rec.get("id") or "rec")
    meta: Dict[str, Any] = {
        "id": rid,
        "title": rec.get("title"),
        "summary": rec.get("summary"),
        "reason": rec.get("reason"),
        "focus": focus,
        "why": _why_line(rec),
    }
    # Optional bookkeeping fields we sometimes attach to recs.
    for key in ("risk", "impact", "effort", "tags"):
        if key in rec:
            meta[key] = rec[key]

    # Propagate acceptance_criteria so edit_file/repair_file prompts can use it.
    if "acceptance_criteria" in rec:
        meta["acceptance_criteria"] = rec["acceptance_criteria"]

    return meta


def _build_target_meta(env: TargetsEnvelope, target_path: str) -> Dict[str, Any]:
    """
    Extract compact target-selection metadata for a specific path from a
    TargetsEnvelope. We keep this small and structured for the edit-file stage.
    """
    path_norm = str(target_path).strip()
    if not path_norm:
        return {}

    for t in env.get("targets", []) or []:
        if not isinstance(t, dict):
            continue
        p = t.get("path")
        if not isinstance(p, str) or p.strip() != path_norm:
            continue

        meta: Dict[str, Any] = {}
        # Core target-selection fields.
        for key in (
            "intent",
            "rationale",
            "success_criteria",
            "effort",
            "risk",
            "dependencies",
        ):
            val = t.get(key)
            if val:
                meta[key] = val

        # Per-file micro-plan / constraints / notes, if present.
        for key in (
            "local_plan",
            "plan",
            "constraints",
            "edit_constraints",
            "notes_for_editor",
            "editor_notes",
            "notes",
        ):
            val = t.get(key)
            if val:
                meta[key] = val

        return meta

    return {}


def _should_generate_edit_for(
    target_meta: Dict[str, Any],
    per_file_analysis: Dict[str, Any],
) -> bool:
    """
    Decide whether we should actually generate an edit for this file.

    Rules:
      - Paths with intent in {"inspect", "context_only"} are never edited.
      - Paths with intent in {"edit", "create"} are eligible for edits.
      - Unknown/missing intent values are treated permissively (like "edit").
      - If analysis.has should_edit explicitly False, skip.
    """
    intent_raw = target_meta.get("intent")
    intent = (intent_raw or "edit").strip().lower()

    # Never generate edits for inspect/context_only
    if intent in ("inspect", "context_only"):
        return False

    # Respect explicit per-file analysis veto
    if isinstance(per_file_analysis, dict) and per_file_analysis.get("should_edit") is False:
        return False

    # For "edit", "create", or any other value, allow edits
    return True


def _apply_unified_patch_with_debug(
    *,
    old_text: str,
    patch_text: str,
    rel_path: str,
    rec_id: str,
    error_cb: Optional[ErrorFn],
    trace_fn: Optional[TraceFn],
) -> Optional[str]:
    """
    Wrapper around apply_unified_patch that emits rich debug information when
    the patch cannot be applied (e.g., unified diff contains no hunks).

    On failure, this logs a short head of the patch plus a simple structural
    flag and returns None so the caller can degrade gracefully.
    """
    base_norm = old_text.replace("\r\n", "\n").replace("\r", "\n")
    patch_str = patch_text or ""
    try:
        return apply_unified_patch(base_norm, patch_str)
    except Exception as e:  # pragma: no cover - defensive
        head_lines = patch_str.splitlines()[:40]

        import re as _re
        
        hunk_re = _re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
        has_hunk = any(hunk_re.match(ln) for ln in head_lines)

        payload: Dict[str, Any] = {
            "error": str(e),
            "path": rel_path,
            "rec_id": rec_id,
            "diff_head": "\n".join(head_lines),
            "has_hunk": has_hunk,
        }

        if error_cb is not None:
            error_cb("apply_patch_preview", payload)

        if trace_fn is not None:
            try:
                trace_fn("EDITS", "apply_patch_preview_error", payload)
            except Exception:
                # Best-effort: never let tracing failures break the pipeline
                pass

        # Structured error log for patch application failure.
        try:
            alog.error("propose_edits.apply_patch_failed", meta=payload, exc_info=True)
        except Exception:
            logging.exception("[propose_edits] apply_patch_failed internal logger error")

        return None


def _refine_targets_envelope(env: TargetsEnvelope) -> TargetsEnvelope:
    """
    Apply any analysis-time refinements (e.g. updated_targets) to the targets
    envelope returned from the target-selection + analysis stages.

    - If env["updated_targets"] is present, we merge it into env["targets"],
      letting per-path fields like local_plan / constraints / notes_for_editor
      override the original entries.

    We also avoid downgrading intent (edit/create -> inspect/context_only).
    """
    try:
        updated = env.get("updated_targets")
    except Exception:
        updated = None

    if not isinstance(updated, list) or not updated:
        # Nothing to refine; ensure the basic keys are present.
        env.setdefault("targets", env.get("targets") or [])
        return env

    def _intent_priority(val: Any) -> int:
        if not isinstance(val, str):
            return 0
        v = val.strip().lower()
        if v == "context_only":
            return 1
        if v == "inspect":
            return 2
        if v in ("edit", "create"):
            return 3
        return 0

    # Build a mapping path -> updated entry for quick lookup.
    updated_by_path: Dict[str, Dict[str, Any]] = {}
    for item in updated:
        if not isinstance(item, dict):
            continue
        p = item.get("path")
        if not isinstance(p, str):
            continue
        key = p.strip()
        if not key:
            continue
        updated_by_path[key] = item

    base_targets = env.get("targets") or []
    if not isinstance(base_targets, list):
        base_targets = []

    merged_targets: List[Dict[str, Any]] = []

    # First pass: merge any existing targets with their updated counterparts.
    for t in base_targets:
        if not isinstance(t, dict):
            continue
        p = t.get("path")
        key = p.strip() if isinstance(p, str) else ""
        if key and key in updated_by_path:
            merged = dict(t)
            updated_entry = updated_by_path[key]

            for k, v in updated_entry.items():
                if k == "path":
                    continue
                if k == "intent":
                    existing_intent = merged.get("intent")
                    if existing_intent is not None and _intent_priority(v) <= _intent_priority(
                        existing_intent
                    ):
                        # Do not downgrade or flip equal-priority intent.
                        continue
                merged[k] = v

            merged_targets.append(merged)
        else:
            merged_targets.append(t)

    # Second pass: add any new targets that only exist in updated_targets.
    existing_paths = {
        str(t.get("path") or "").strip()
        for t in merged_targets
        if isinstance(t, dict)
    }
    for key, updated_entry in updated_by_path.items():
        if key not in existing_paths:
            merged_targets.append(updated_entry)

    env["targets"] = merged_targets
    return env


def _normalize_targets_envelope(value: Any) -> TargetsEnvelope:
    """
    Normalize a per-rec 'targets' value into a TargetsEnvelope-like dict.

    Supports:
      - New-style envelopes: { "targets": [...], "selected_files": [...], ... }
      - Legacy lists of paths: ["a.js", "b.js", ...]
      - Anything else -> empty envelope.
    """
    if isinstance(value, dict):
        env = dict(value)
        env.setdefault("targets", env.get("targets") or [])
        env.setdefault("selected_files", env.get("selected_files") or [])
        env.setdefault("llm_payload_preview", env.get("llm_payload_preview") or [])
        return _refine_targets_envelope(env)

    if isinstance(value, list):
        paths = [p for p in value if isinstance(p, str) and p.strip()]
        env = {
            # NOTE: we only reliably have the path here; other target fields are
            # meaningful only when the model returns a full envelope.
            "targets": [{"path": p} for p in paths],
            "selected_files": paths,
            "llm_payload_preview": [],
        }
        return _refine_targets_envelope(env)

    env = {
        "targets": [],
        "selected_files": [],
        "llm_payload_preview": [],
    }
    return _refine_targets_envelope(env)


def _is_glob_like(p: str) -> bool:
    return any(ch in p for ch in ("*", "?", "["))


def _extract_target_paths(env: TargetsEnvelope) -> List[str]:
    paths: List[str] = []
    for t in env.get("targets", []):
        if isinstance(t, dict):
            p = t.get("path")
            if isinstance(p, str):
                p = p.strip()
                if not p or _is_glob_like(p):
                    continue
                paths.append(p)
    return paths


def _generate_and_collect_edit_for_file(
    *,
    file_ctx: FileEditContext,
    chat_json_fn: ChatJsonFn,
    timeout_runner: TimeoutRunner,
    timeout_sentinel: Any,
    timeout_sec: float,
    error_cb: Optional[ErrorFn],
    trace_fn: Optional[TraceFn],
    brief_hash: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Core helper that runs generate_edits_for_file for a single file, applies
    patches (for preview), and returns:

        proposed_entry, edit_obj

    Both are None if the edit could not be produced.

    The FileEditContext bundles all per-file state we want to show the model:
    the current file content, structured per-file analysis, and compact
    rec/target metadata.
    """
    # Local import to avoid circular import issues at module import time.
    from .generate_edits import generate_edits_for_file  # type: ignore

    rid = str(file_ctx.rec_meta.get("id") or "rec")
    rel_path = file_ctx.path
    why = (file_ctx.rec_meta.get("why") or "").strip()
    old_text = file_ctx.content

    has_analysis = bool(file_ctx.analysis)

    # Structured start log for this per-file generate step.
    try:
        alog.info("propose_edits.generate_edit_start", meta={
            "rec_id": rid,
            "path": rel_path,
            "has_analysis": has_analysis,
        })
    except Exception:
        # Fallback to stdlib logging if structured logger has issues.
        logging.info(
            "[propose_edits] generate_edit start rec_id=%s path=%s has_analysis=%s",
            rid,
            rel_path,
            has_analysis,
        )

    # Compute a per-call model_override from the FileEditContext if present.
    model_override: Optional[str] = None
    try:
        # Prefer rec_meta overrides, then target_meta overrides.
        for key in ("model", "model_override", "llm_model"):
            v = None
            if isinstance(file_ctx.rec_meta, dict):
                v = file_ctx.rec_meta.get(key)
            if (not v or not isinstance(v, str) or not v.strip()) and isinstance(getattr(file_ctx, "target_meta", None), dict):
                v = file_ctx.target_meta.get(key)
            if isinstance(v, str) and v.strip():
                model_override = v.strip()
                break
    except Exception:
        model_override = None

    # If no explicit model_override was provided by rec/target metadata, allow a
    # runtime fallback to the configured model for this logical stage. This
    # keeps stage-aware model resolution centralized in config.get_model_for.
    if not model_override:
        try:
            # Import lazily to avoid import cycles during module import.
            from .. import config

            getter = getattr(config, "get_model_for", None)
            if callable(getter):
                resolved = getter(STAGE_LABEL)
                if isinstance(resolved, str) and resolved.strip():
                    model_override = resolved.strip()
                    try:
                        alog.info(
                            "propose_edits.resolved_model",
                            meta={"rec_id": rid, "path": rel_path, "resolved_model": model_override},
                        )
                    except Exception:
                        logging.info(
                            "[propose_edits] resolved model for rec_id=%s path=%s => %s",
                            rid,
                            rel_path,
                            model_override,
                        )
        except Exception:
            # Best-effort only; do not fail the edit generation if config is
            # unavailable or throws.
            model_override = model_override

    # Decide whether to call the generator in an "edit" or "repair" mode.
    # Historically this switched the stage string passed to generate_edits_for_file
    # (e.g. "propose_repairs"). To avoid tightly coupling stage labels across
    # modules we keep the stage param stable (STAGE_LABEL) and instead set an
    # explicit routing hint on the FileEditContext. Consumers of file_ctx
    # (including generate_edits_for_file) should look at file_ctx.routing_reason
    # (or file_ctx.repair_mode if present) to decide whether the call is a
    # repair-specific invocation. This keeps routing explicit and avoids
    # breaking callers that expect the canonical stage label.
    stage_for_generate = STAGE_LABEL
    try:
        def _is_repair_from(d: Any) -> bool:
            if not isinstance(d, dict):
                return False
            if d.get("repair") is True:
                return True
            if d.get("is_repair") is True:
                return True
            m = d.get("mode")
            if isinstance(m, str) and m.strip().lower() == "repair":
                return True
            it = d.get("intent")
            if isinstance(it, str) and it.strip().lower() == "repair":
                return True
            return False

        if _is_repair_from(getattr(file_ctx, "analysis", {})) or _is_repair_from(getattr(file_ctx, "target_meta", {})) or _is_repair_from(getattr(file_ctx, "rec_meta", {})):
            # Instead of changing the stage string, record an explicit routing
            # hint on the FileEditContext. generate_edits_for_file implementations
            # should consult file_ctx.routing_reason == 'repair' when they
            # need to switch to repair-specific prompts/validation.
            try:
                file_ctx.routing_reason = "repair"
            except Exception:
                # Best-effort only; non-fatal if we cannot mutate the context.
                pass
            try:
                alog.info("propose_edits.marked_repair_routing", meta={"rec_id": rid, "path": rel_path, "routing": "repair"})
            except Exception:
                logging.info("[propose_edits] marked routing_reason=repair for rec_id=%s path=%s", rid, rel_path)
    except Exception:
        # Best-effort only; keep default stage on any failure.
        stage_for_generate = STAGE_LABEL

    def _gen() -> Optional[Dict[str, Any]]:  # type: ignore[return-type]
        # Pass stage (kept stable) and any discovered model_override to the generator.
        # The generator may consult file_ctx.routing_reason to alter behavior.
        return generate_edits_for_file(
            file_ctx=file_ctx,
            chat_json_fn=chat_json_fn,
            stage=stage_for_generate,
            model_override=model_override,
        )

    start_ts = time.time()
    try:
        # Structured timeout-runner start log.
        try:
            alog.info("propose_edits.timeout_runner_start", meta={
                "rec_id": rid,
                "path": rel_path,
                "timeout_sec": float(timeout_sec),
            })
        except Exception:
            logging.info(
                "[propose_edits] timeout_runner start rec_id=%s path=%s timeout_sec=%.1f",
                rid,
                rel_path,
                timeout_sec,
            )

        res = timeout_runner(_gen, timeout_sec, desc=f"generate_edit:{rel_path}")
        elapsed = time.time() - start_ts

        # Structured timeout-runner done log.
        try:
            alog.info("propose_edits.timeout_runner_done", meta={
                "rec_id": rid,
                "path": rel_path,
                "elapsed_s": float(elapsed),
            })
        except Exception:
            logging.info(
                "[propose_edits] timeout_runner done rec_id=%s path=%s elapsed=%.2fs",
                rid,
                rel_path,
                elapsed,
            )
    except Exception as e:
        elapsed = time.time() - start_ts
        # Preserve original stack trace information and emit structured error.
        try:
            alog.error(
                "propose_edits.timeout_runner_exception",
                meta={
                    "reason": "timeout_runner_exception",
                    "path": rel_path,
                    "rec_id": rid,
                    "seconds": float(timeout_sec),
                    "elapsed_s": float(elapsed),
                    "error": str(e),
                },
                exc_info=True,
            )
        except Exception:
            logging.exception(
                "[propose_edits] timeout_runner ERROR rec_id=%s path=%s elapsed=%.2fs",
                rid,
                rel_path,
                elapsed,
            )
        if error_cb:
            error_cb(
                "generate_edit",
                {
                    "reason": "timeout_runner_exception",
                    "path": rel_path,
                    "rec_id": rid,
                    "seconds": timeout_sec,
                },
            )
        return None, None

    if res is timeout_sentinel:
        # Structured timeout log for sentinel case.
        try:
            alog.warning("propose_edits.timeout", meta={
                "reason": "timeout",
                "path": rel_path,
                "rec_id": rid,
                "seconds": float(timeout_sec),
            })
        except Exception:
            logging.warning(
                "[propose_edits] timeout rec_id=%s path=%s timeout_sec=%.1f",
                rid,
                rel_path,
                timeout_sec,
            )
        if error_cb:
            error_cb(
                "generate_edit",
                {
                    "reason": "timeout",
                    "path": rel_path,
                    "rec_id": rid,
                    "seconds": timeout_sec,
                },
            )
        return None, None

    edit_obj = res
    if not edit_obj:
        # Structured warning for malformed/empty LLM edit response.
        try:
            alog.warning(
                "propose_edits.no_edit",
                meta={
                    "reason": "no_edit",
                    "path": rel_path,
                    "rec_id": rid,
                    "message": "[LLM] Malformed or empty edit response; skipping this file for the recommendation.",
                },
            )
        except Exception:
            logging.warning(
                "[propose_edits] no_edit rec_id=%s path=%s (empty or malformed edit)",
                rid,
                rel_path,
            )
        if error_cb:
            error_cb(
                "generate_edit",
                {
                    "reason": "no_edit",
                    "path": rel_path,
                    "rec_id": rid,
                    "message": "[LLM] Malformed or empty edit response; skipping this file for the recommendation.",
                },
            )
        return None, None

    # Structured raw edit info
    try:
        alog.info(
            "propose_edits.raw_edit",
            meta={
                "rec_id": rid,
                "target_path": rel_path,
                "edit_keys": sorted(list(edit_obj.keys())),
                "edit_path": edit_obj.get("path"),
            },
        )
    except Exception:
        logging.info(
            "[propose_edits] raw_edit rec_id=%s target_path=%s edit_keys=%s edit_path=%s",
            rid,
            rel_path,
            sorted(edit_obj.keys()),
            edit_obj.get("path"),
        )

    preview_new: Optional[str] = None

    # Determine the baseline for applying patches.
    #
    # POLICY:
    #   - Normal (non-repair) edits MUST apply patches against the exact full file snapshot
    #     we read from disk (old_text). Do NOT use preview/snippet content here.
    #   - Repair routing may legitimately use a preview/snippet baseline if the model
    #     was only shown that baseline during repair.
    base_for_patch = old_text
    base_for_patch_source = "file_current"
    is_repair_routing = (getattr(file_ctx, "routing_reason", None) == "repair")

    if is_repair_routing:
        try:
            if isinstance(getattr(file_ctx, "analysis", None), dict):
                for k in ("preview_content", "preview", "base_preview", "base_content", "base"):
                    v = file_ctx.analysis.get(k)
                    if isinstance(v, str) and v:
                        base_for_patch = v
                        base_for_patch_source = f"analysis:{k}"
                        break
        except Exception:
            base_for_patch = old_text
            base_for_patch_source = "file_current"

    # Allow the generator to explicitly supply a patch base ONLY in repair routing.
    # In normal mode, accepting a patch_base would violate the invariant that patches
    # apply to the exact full snapshot the model saw.
    if is_repair_routing:
        try:
            for k in ("patch_base", "base", "base_content", "base_preview", "preview_base", "apply_to", "preview_content"):
                if k in edit_obj:
                    v = edit_obj.get(k)
                    if isinstance(v, str) and v:
                        base_for_patch = v
                        base_for_patch_source = f"edit_obj:{k}"
                        break
        except Exception:
            pass

    # Optional observability: record when we used a non-canonical baseline (repair only).
    if is_repair_routing and base_for_patch is not old_text and trace_fn is not None:
        try:
            trace_fn(
                "EDITS",
                "patch_baseline_selected",
                {
                    "rec_id": rid,
                    "path": rel_path,
                    "baseline_source": base_for_patch_source,
                    "baseline_len": int(len(base_for_patch or "")),
                    "file_current_len": int(len(old_text or "")),
                },
            )
        except Exception:
            pass

    # --- Tiered behavior: prefer applying patch (patch_unified/patch) first,
    # then fall back to returning full 'content' only if patch application
    # failed. This enables patch-first workflows while preserving legacy full
    # content outputs as a fallback.
    patch_applied = False
    patch_attempted = False
    patch_failed = False

    try:
        # Try unified patch first if present
        if "patch_unified" in edit_obj and isinstance(edit_obj.get("patch_unified"), str):
            patch_attempted = True
            applied = _apply_unified_patch_with_debug(
                old_text=base_for_patch,
                patch_text=edit_obj["patch_unified"],
                rel_path=rel_path,
                rec_id=rid,
                error_cb=error_cb,
                trace_fn=trace_fn,
            )
            if applied is not None:
                preview_new = applied
                patch_applied = True
            else:
                patch_failed = True

        # If we didn't apply yet, try legacy 'patch' key and normalize it.
        if not patch_applied and "patch" in edit_obj and isinstance(edit_obj.get("patch"), str):
            patch_attempted = True
            applied = _apply_unified_patch_with_debug(
                old_text=base_for_patch,
                patch_text=edit_obj["patch"],
                rel_path=rel_path,
                rec_id=rid,
                error_cb=error_cb,
                trace_fn=trace_fn,
            )
            if applied is not None:
                preview_new = applied
                patch_applied = True
                # Normalize outward-facing field name so downstream consumers
                # observe 'patch_unified' consistently.
                try:
                    edit_obj["patch_unified"] = edit_obj.pop("patch")
                except Exception:
                    pass
            else:
                patch_failed = True

        # If patch application(s) failed (or none provided/applied) and the LLM
        # provided a full-file 'content', fall back to that.
        if not patch_applied and "content" in edit_obj and isinstance(edit_obj.get("content"), str):
            preview_new = edit_obj["content"]
            if patch_failed:
                # Emit a warning so operators/tests can detect that we fell back
                # to full content after a patch apply failure.
                payload = {"rec_id": rid, "path": rel_path, "note": "patch_apply_failed_fell_back_to_content"}
                try:
                    alog.warning("propose_edits.patch_apply_failed_fallback_content", meta=payload)
                except Exception:
                    logging.warning(
                        "[propose_edits] patch apply failed; falling back to content for rec_id=%s path=%s",
                        rid,
                        rel_path,
                    )
                if error_cb:
                    error_cb("generate_edit", {"reason": "patch_apply_failed_fallback_to_content", "path": rel_path, "rec_id": rid})
    except Exception:
        # Defensive: any unexpected failure in tiered resolution should not
        # break the pipeline; continue with whatever preview_new we have.
        pass

    # Ensure rec_id always present
    if "rec_id" not in edit_obj:
        edit_obj["rec_id"] = rid

    # Path sanity check:
    raw_path = edit_obj.get("path")
    raw_path = raw_path.strip() if isinstance(raw_path, str) else ""

    # Normalize for safe prefix checks
    target_norm = rel_path.replace("\\", "/").rstrip("/")
    raw_norm = raw_path.replace("\\", "/").rstrip("/")

    effective_path = rel_path

    if raw_path and raw_path != rel_path:
        # Allow redirect to a child path under the target "directory"
        if target_norm and raw_norm.startswith(target_norm + "/"):
            effective_path = raw_path
            edit_obj["path"] = effective_path
            try:
                alog.info("propose_edits.accepted_child_edit_path", meta={
                    "rec_id": rid,
                    "target": rel_path,
                    "edit_path": raw_path,
                })
            except Exception:
                logging.info(
                    "[propose_edits] accepted child edit path rec_id=%s target=%s edit_path=%s",
                    rid,
                    rel_path,
                    raw_path,
                )
        else:
            try:
                alog.warning(
                    "propose_edits.edit_path_mismatch",
                    meta={
                        "rec_id": rid,
                        "target": rel_path,
                        "edit_path": raw_path,
                    },
                )
            except Exception:
                logging.warning(
                    "[propose_edits] edit path mismatch rec_id=%s target=%s edit_path=%s; forcing edit.path to target.",
                    rid,
                    rel_path,
                    raw_path,
                )
            edit_obj["path"] = rel_path
    elif not raw_path:
        edit_obj["path"] = rel_path

    # From this point on, prefer the effective path for preview metadata/diffs.
    rel_path = edit_obj.get("path") or rel_path

    # Optional human-readable summary coming from system.edit_file
    summary_text = ""
    if isinstance(edit_obj.get("summary"), str):
        summary_text = edit_obj["summary"].strip()

    preview_bytes = len((preview_new or "").encode("utf-8"))

    # --- New block: normalize attempt metadata and decide last_output_type ---
    # Determine last_output_type based on what we actually used to build preview_new.
    last_output_type = "unknown"
    try:
        # Normalize attempt metadata if present in edit_obj under common keys.
        attempts_list: Optional[List[Any]] = None
        for key in ("attempts", "attempt_history", "edit_attempts", "attempt_meta"):
            if key in edit_obj:
                val = edit_obj.get(key)
                if isinstance(val, list):
                    attempts_list = val
                elif isinstance(val, dict):
                    attempts_list = [val]
                elif val is None:
                    attempts_list = []
                else:
                    # Coerce single scalar value into a single-item list for visibility.
                    attempts_list = [val]
                # Normalize onto edit_obj['attempts'] so downstream consumers see a
                # consistent field regardless of source key used by generate_edits.
                try:
                    edit_obj["attempts"] = attempts_list
                except Exception:
                    # Best-effort only; do not break on inability to write back.
                    pass
                break

        # Decide last_output_type from the tiered resolution results.
        if patch_applied:
            last_output_type = "patch_unified"
        else:
            # If we fell back to full content, prefer 'content' when present.
            if preview_new is not None and "content" in edit_obj and isinstance(edit_obj.get("content"), str) and preview_new == edit_obj.get("content"):
                last_output_type = "content"
            elif "patch_unified" in edit_obj and isinstance(edit_obj.get("patch_unified"), str):
                # If a patch was provided but not applied, still surface that the
                # generator provided a patch (even though it failed to apply).
                last_output_type = "patch_unified"
            elif "content" in edit_obj and isinstance(edit_obj.get("content"), str):
                last_output_type = "content"
            else:
                last_output_type = "unknown"
    except Exception:
        last_output_type = "unknown"

    # Unit-test hint: proposed_entry should include 'attempts' (list) and
    # 'last_output_type' when generate_edits supplies attempt metadata.
    # Tests may assert on these keys for auditability.
    # --- end new block ---

    # detect create-mode hints and propagate. Prefer explicit flags from the
    # generate_edits_for_file implementation; fall back to target_meta.intent == 'create'.
    # NOTE: Do not mark is_new solely because the file was empty on disk.
    is_new_hint = False
    try:
        is_new_hint = bool(
            edit_obj.get("is_new") is True
            or edit_obj.get("create_mode")
            or edit_obj.get("create_prompt")
            or (isinstance(getattr(file_ctx, "target_meta", None), dict) and file_ctx.target_meta.get("intent") == "create")
        )
    except Exception:
        is_new_hint = False

    if is_new_hint:
        # Ensure downstream consumers always see the explicit boolean marker.
        edit_obj["is_new"] = True
        try:
            alog.info("propose_edits.marked_new_file", meta={"rec_id": rid, "path": rel_path, "is_new": True})
        except Exception:
            logging.info("[propose_edits] marked new file rec_id=%s path=%s", rid, rel_path)
        # Unit-test hint: tests should assert returned edit_obj contains is_new==True
        # and proposed_entry.diff shows additions (old -> new). Keep detection here
        # so the generated diff_text below already reflects the full-file content.

    # Build unified diff for preview. For new files, produce a diff whose old
    # filename header is /dev/null so tools and tests recognize this as a file
    # addition and the diff shows the entire new file content as additions.
    # Using '/dev/null' in the left header mirrors how git represents added files
    # and makes UI/approval logic clearer to future editors.
    diff_text = ""
    if preview_new is not None:
        is_new_final = bool(edit_obj.get("is_new") is True or is_new_hint)
        # Choose baseline_for_diff: for new files it's empty; otherwise it's the
        # same baseline we used to apply the patch preview.
        # In normal mode this is the full on-disk snapshot (old_text). In repair
        # routing it may be a preview/snippet baseline.
        baseline_for_diff = "" if is_new_final else base_for_patch
        if is_new_final:
            # Old text should be treated as empty for new-file diffs.
            diff_text = generate_unified_diff(
                "/dev/null",
                f"b/{rel_path}",
                "",
                preview_new,
            )
        else:
            diff_text = generate_unified_diff(
                f"a/{rel_path}",
                f"b/{rel_path}",
                baseline_for_diff,
                preview_new,
            )

    # Expose preview content as a stable repair baseline for downstream stages.
    if preview_new is not None:
        try:
            edit_obj["preview_content"] = preview_new
        except Exception:
            # Best-effort only; do not break if we cannot attach.
            pass

    # Reject no-op or header-only diffs: if not a new file and either the preview
    # equals the existing content (no-op) or the unified diff contains no hunks,
    # treat as a non-actionable edit and skip returning it.
    try:
        if preview_new is not None and not (edit_obj.get("is_new") is True or is_new_hint):
            # Compare against the same baseline we applied the patch preview to.
            # In normal mode this equals the full on-disk snapshot; in repair routing
            # it may be a preview/snippet baseline.
            noop = preview_new == base_for_patch
            lacks_hunk = "@@" not in (diff_text or "")
            if noop or lacks_hunk:
                reason = "no_op" if noop else "header_only_diff"
                payload = {"rec_id": rid, "path": rel_path, "reason": reason}
                try:
                    alog.warning("propose_edits.rejected_noop_or_header_only", meta=payload)
                except Exception:
                    logging.warning(
                        "[propose_edits] rejected no-op/header-only diff rec_id=%s path=%s reason=%s",
                        rid,
                        rel_path,
                        reason,
                    )
                if error_cb:
                    error_cb("generate_edit", {"reason": reason, "path": rel_path, "rec_id": rid})
                return None, None
    except Exception:
        # Any failure in this defensive check should not break the pipeline;
        # continue gracefully.
        pass

    proposed_entry: Dict[str, Any] = {
        "rec_id": rid,
        "path": rel_path,
        "diff": diff_text,
        "preview_bytes": preview_bytes,
        "preview_content": preview_new,
        "why": why or _why_line(file_ctx.rec_meta) if hasattr(file_ctx, "rec_meta") else why,
        "summary": summary_text or None,
    }

    # Attach is_new flag to the proposed preview entry for UI/tests visibility.
    if is_new_hint:
        proposed_entry["is_new"] = True

    # Surface structured per-file analysis (plan/constraints/notes) for UI/tests.
    if file_ctx.analysis:
        proposed_entry["analysis"] = file_ctx.analysis

    # Extract tolerant model metadata from edit_obj and propagate it.
    chosen_model: Optional[str] = None
    try:
        # Try structured 'llm' block first.
        llm_block = edit_obj.get("llm")
        if isinstance(llm_block, dict):
            m = llm_block.get("model")
            if isinstance(m, str) and m.strip():
                chosen_model = m.strip()

        if not chosen_model:
            # Fallbacks: 'model' or 'model_used'
            for key in ("model", "model_used"):
                m2 = edit_obj.get(key)
                if isinstance(m2, str) and m2.strip():
                    chosen_model = m2.strip()
                    break
    except Exception:
        chosen_model = None

    if chosen_model:
        # Do not overwrite an existing explicit edit_obj['model'] value.
        if "model" not in edit_obj:
            edit_obj["model"] = chosen_model
        # Also attach to the proposed preview so UIs can show it.
        proposed_entry["model"] = chosen_model

    # Propagate normalized attempt metadata and last_output_type into the
    # proposed entry so UIs and tests can show attempt history and final
    # output type (e.g., 'patch_unified' vs 'content'). This mirrors the
    # normalization we applied onto edit_obj['attempts'] above.

    if "attempts" in edit_obj and isinstance(edit_obj.get("attempts"), list):
        try:
            proposed_entry["attempts"] = edit_obj.get("attempts")
        except Exception:
            # Best-effort only; do not fail on inability to attach.
            pass

    proposed_entry["last_output_type"] = last_output_type

    if trace_fn:
        try:
            trace_payload: Dict[str, Any] = {
                "rec_id": rid,
                "path": rel_path,
                "has_content": bool("content" in edit_obj),
                "has_patch": bool("patch_unified" in edit_obj),
                "brief_hash": brief_hash,
            }
            if chosen_model:
                trace_payload["model"] = chosen_model
            # Add attempt metadata to the trace payload for observability.
            if isinstance(edit_obj.get("attempts"), list):
                trace_payload["attempt_count"] = int(len(edit_obj.get("attempts")))
            trace_payload["last_output_type"] = last_output_type
            # Include the explicit routing hint so traces show whether this
            # call was intended as a 'repair' routing.
            if getattr(file_ctx, "routing_reason", None):
                trace_payload["routing_reason"] = file_ctx.routing_reason
            trace_fn(
                "EDITS",
                "llm",
                trace_payload,
            )
        except Exception:  # pragma: no cover - defensive
            try:
                alog.error(
                    "propose_edits.trace_fn_failed",
                    meta={"rec_id": rid, "path": rel_path},
                    exc_info=True,
                )
            except Exception:
                logging.exception(
                    "[propose_edits] trace_fn failed for rec_id=%s path=%s",
                    rid,
                    rel_path,
                )

    return proposed_entry, edit_obj


def propose_edits_for_single_recommendation(
    *,
    root: Path,
    rec: Dict[str, Any],
    targets: List[str],
    focus: str,
    chat_json_fn: ChatJsonFn,
    edit_schema: JsonSchema,  # kept for signature compatibility; no longer used here
    timeout_runner: TimeoutRunner,
    timeout_sentinel: Any,
    timeout_sec: float,
    should_cancel: Callable[[], bool],
    progress_cb: Optional[ProgressFn] = None,
    error_cb: Optional[ErrorFn] = None,
    trace_fn: Optional[TraceFn] = None,
    brief_hash: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate proposed edits for a single recommendation and its target files.

    This helper still accepts a legacy List[str] of target paths and is used by
    simpler pipelines. The orchestrated multi-rec path goes through
    propose_edits_for_recommendations() which now understands full
    TargetsEnvelope objects.

    NOTE: The edit_schema parameter is kept for backward-compatible call sites
    but validation is now handled inside generate_edits_for_file / schemas.
    """
    proposed: List[Dict[str, Any]] = []
    edits: List[Dict[str, Any]] = []

    rec_meta = _build_rec_meta(rec, focus)
    rid = rec_meta["id"]
    because = rec_meta["why"]
    targets = targets or []

    total_files = max(1, len(targets))
    done_files = 0

    for rel_path in targets:
        if should_cancel():
            try:
                alog.warning(
                    "propose_edits.cancellation_requested",
                    meta={"rec_id": rid, "path": rel_path},
                )
            except Exception:
                logging.warning(
                    "[propose_edits_single] cancellation requested; stopping at rec_id=%s path=%s",
                    rid,
                    rel_path,
                )
            break

        old_text = _read_file_text_if_exists(root, rel_path) or ""

        if progress_cb:
            progress_cb(
                "generate_edits_iter",
                {
                    "rec_id": rid,
                    "path": rel_path,
                    "why": because,
                    "idx": done_files + 1,
                    "total": total_files,
                },
            )

        file_ctx = FileEditContext(
            path=rel_path,
            language=_guess_language_from_path(rel_path),
            content=old_text,
            analysis={},  # legacy path: no per-file analysis available
            rec_meta=rec_meta,
            target_meta={},  # legacy path: no targets envelope available
        )

        proposed_entry, edit_obj = _generate_and_collect_edit_for_file(
            file_ctx=file_ctx,
            chat_json_fn=chat_json_fn,
            timeout_runner=timeout_runner,
            timeout_sentinel=timeout_sentinel,
            timeout_sec=timeout_sec,
            error_cb=error_cb,
            trace_fn=trace_fn,
            brief_hash=brief_hash,
        )

        if proposed_entry is None or edit_obj is None:
            try:
                alog.warning("propose_edits.skipped_file", meta={"rec_id": rid, "path": rel_path})
            except Exception:
                logging.warning(
                    "[propose_edits] skipped file rec_id=%s path=%s (no proposed edit)",
                    rid,
                    rel_path,
                )
            # Error already logged / reported via callbacks.
            continue

        done_files += 1

        proposed.append(proposed_entry)
        edits.append(edit_obj)

        if progress_cb:
            progress_cb(
                "edit_done",
                {
                    "rec_id": rid,
                    "path": rel_path,
                },
            )

    return proposed, edits


def propose_edits_for_recommendations(
    *,
    root: Path,
    recs: Sequence[Dict[str, Any]],
    rec_targets: Dict[str, Any],
    focus: str,
    chat_json_fn: ChatJsonFn,
    edit_schema: JsonSchema,  # kept for signature compatibility; no longer used here
    timeout_runner: TimeoutRunner,
    timeout_sentinel: Any,
    timeout_sec: float,
    should_cancel: Callable[[], bool],
    progress_cb: Optional[ProgressFn] = None,
    error_cb: Optional[ErrorFn] = None,
    trace_fn: Optional[TraceFn] = None,
    brief_hash: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each (recommendation, target file) pair, build a FileEditContext,
    call generate_edits_for_file, apply the patch to build a preview, and
    return:

        proposed: [
            {
            rec_id,
            path,
            diff,
            preview_bytes,
            preview_content,  # full candidate file content (if available)
            why,
            summary?,
            analysis?,        # structured per-file analysis (if available)
            },
            ...
        ]
        edits:    [{path, rec_id, content? | patch_unified?}, ...]

    Only targets whose intent == "edit" and whose per-file analysis does not
    explicitly set should_edit == False will be sent to the edit LLM. Targets
    with intent in {"inspect", "context_only", ...} remain analyze-only and
    still contribute cross-file notes and updated_targets.
    """
    proposed: List[Dict[str, Any]] = []
    edits: List[Dict[str, Any]] = []

    normalized_envs: Dict[str, TargetsEnvelope] = {}

    def _env_for(rid: str) -> TargetsEnvelope:
        raw = rec_targets.get(rid) or {}
        if rid in normalized_envs:
            return normalized_envs[rid]
        env = _normalize_targets_envelope(raw)
        normalized_envs[rid] = env
        return env

    # First pass: compute total_files (only those we will actually attempt to edit)
    total_files = 0
    for rec in recs:
        rid = str(rec.get("id") or "rec")
        env = _env_for(rid)
        target_paths = _extract_target_paths(env)
        if not target_paths:
            continue

        analysis_block = (
            rec.get("_analysis")
            or rec.get("_target_analysis")
            or {}
        )
        per_file_raw = (
            analysis_block.get("per_file")
            or analysis_block.get("analysis_by_path")
            or {}
        )
        per_file_analysis_map: Dict[str, Dict[str, Any]] = {}

        if isinstance(per_file_raw, dict):
            for k, v in per_file_raw.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                norm_k = k.replace("\\", "/")
                per_file_analysis_map[norm_k] = v

        for rel_path in target_paths:
            norm_rel = str(rel_path).replace("\\", "/")
            per_file_analysis = (
                per_file_analysis_map.get(norm_rel)
                or per_file_analysis_map.get(str(rel_path))
                or {}
            )

            target_meta = _build_target_meta(env, rel_path)
            if _should_generate_edit_for(target_meta, per_file_analysis):
                total_files += 1

    try:
        alog.info("propose_edits.start", meta={"recs": len(recs), "total_files": total_files})
    except Exception:
        logging.info(
            "[propose_edits] start recs=%d total_files=%d",
            len(recs),
            total_files,
        )

    done_files = 0

    for rec in recs:
        if should_cancel():
            try:
                alog.warning("propose_edits.cancellation_before_rec_loop")
            except Exception:
                logging.warning(
                    "[propose_edits] cancellation requested before rec loop; stopping."
                )
            break

        rec_meta = _build_rec_meta(rec, focus)
        rid = rec_meta["id"]
        because = rec_meta["why"]

        # Normalise this rec's targets envelope.
        env = _env_for(rid)
        target_paths = _extract_target_paths(env)

        # Pull per-file analysis + cross-file notes, if present.
        analysis_block = (
            rec.get("_analysis")
            or rec.get("_target_analysis")
            or {}
        )
        per_file_raw = (
            analysis_block.get("per_file")
            or analysis_block.get("analysis_by_path")
            or {}
        )
        per_file_analysis_map: Dict[str, Dict[str, Any]] = {}

        if isinstance(per_file_raw, dict):
            for k, v in per_file_raw.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                norm_k = k.replace("\\", "/")
                per_file_analysis_map[norm_k] = v

        # Canonical name is analysis_cross_file_notes; keep cross_file_notes as a fallback.
        cross_file_notes = (
            analysis_block.get("analysis_cross_file_notes")
            or analysis_block.get("cross_file_notes")
        )

        try:
            alog.info("propose_edits.rec", meta={"rec_id": rid, "targets": target_paths, "why": because})
        except Exception:
            logging.info(
                "[propose_edits] rec_id=%s targets=%s why=%s",
                rid,
                target_paths,
                because,
            )

        # IMPORTANT: target_paths is already ordered according to the
        # LLM's planning in system.target_select. We preserve this order
        # so that edits proceed from more foundational files to more
        # dependent ones.
        for rel_path in target_paths:
            if should_cancel():
                try:
                    alog.warning(
                        "propose_edits.cancellation_during_rec", meta={"rec_id": rid}
                    )
                except Exception:
                    logging.warning(
                        "[propose_edits] cancellation requested during rec_id=%s; "
                        "stopping further targets.",
                        rid,
                    )
                break

            norm_rel = str(rel_path).replace("\\", "/")
            per_file_analysis = (
                per_file_analysis_map.get(norm_rel)
                or per_file_analysis_map.get(str(rel_path))
                or {}
            )

            # Attach cross-file notes once so the per-file analysis is complete.
            if (
                cross_file_notes
                and isinstance(per_file_analysis, dict)
                and "cross_file_notes" not in per_file_analysis
                and "analysis_cross_file_notes" not in per_file_analysis
            ):
                # Clone so we don't mutate the shared map.
                per_file_analysis = dict(per_file_analysis)
                # Per-file view keeps both names for convenience.
                per_file_analysis["analysis_cross_file_notes"] = cross_file_notes
                per_file_analysis["cross_file_notes"] = cross_file_notes

            target_meta = _build_target_meta(env, rel_path)

            if not _should_generate_edit_for(target_meta, per_file_analysis):
                try:
                    alog.info(
                        "propose_edits.skip_edit",
                        meta={
                            "rec_id": rid,
                            "path": rel_path,
                            "intent": target_meta.get("intent"),
                            "should_edit": per_file_analysis.get("should_edit") if isinstance(per_file_analysis, dict) else None,
                        },
                    )
                except Exception:
                    logging.info(
                        "[propose_edits] skip_edit rec_id=%s path=%s intent=%s should_edit=%s",
                        rid,
                        rel_path,
                        target_meta.get("intent"),
                        per_file_analysis.get("should_edit") if isinstance(per_file_analysis, dict) else None,
                    )
                if progress_cb:
                    try:
                        progress_cb(
                            "edit_skipped",
                            {
                                "rec_id": rid,
                                "path": rel_path,
                                "intent": target_meta.get("intent"),
                                "should_edit": per_file_analysis.get("should_edit")
                                if isinstance(per_file_analysis, dict)
                                else None,
                            },
                        )
                    except Exception:
                        try:
                            alog.error(
                                "propose_edits.progress_cb_failed",
                                meta={"rec_id": rid, "path": rel_path},
                                exc_info=True,
                            )
                        except Exception:
                            logging.exception(
                                "[propose_edits] progress_cb failed for edit_skipped rec_id=%s path=%s",
                                rid,
                                rel_path,
                            )
                continue

            old_text = _read_file_text_if_exists(root, rel_path) or ""

            if progress_cb:
                progress_cb(
                    "generate_edits_iter",
                    {
                        "rec_id": rid,
                        "path": rel_path,
                        "why": because,
                        "idx": done_files + 1,
                        "total": max(1, total_files),
                    },
                )

            done_files += 1

            language = (
                per_file_analysis.get("language")
                if isinstance(per_file_analysis, dict) and per_file_analysis.get("language")
                else _guess_language_from_path(rel_path)
            )

            file_ctx = FileEditContext(
                path=rel_path,
                language=language,
                content=old_text,
                analysis=per_file_analysis if isinstance(per_file_analysis, dict) else {},
                rec_meta=rec_meta,
                target_meta=target_meta,
            )

            proposed_entry, edit_obj = _generate_and_collect_edit_for_file(
                file_ctx=file_ctx,
                chat_json_fn=chat_json_fn,
                timeout_runner=timeout_runner,
                timeout_sentinel=timeout_sentinel,
                timeout_sec=timeout_sec,
                error_cb=error_cb,
                trace_fn=trace_fn,
                brief_hash=brief_hash,
            )

            if proposed_entry is None or edit_obj is None:
                # Error already logged / reported via callbacks.
                continue

            proposed.append(proposed_entry)
            edits.append(edit_obj)

            if progress_cb:
                progress_cb(
                    "edit_done",
                    {
                        "rec_id": rid,
                        "path": rel_path,
                    },
                )

    try:
        alog.info("propose_edits.done", meta={"proposed": len(proposed), "edits": len(edits)})
    except Exception:
        logging.info(
            "[propose_edits] done proposed=%d edits=%d",
            len(proposed),
            len(edits),
        )
    return proposed, edits
