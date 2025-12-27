# aidev/stages/apply_and_refresh.py
# fmt: off
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Lazy imports and small helpers require OS-level operations
import os
import glob
import hashlib

# Prefer not to import the whole aidev package at module import time; default to stdlib logging.
# This avoids importing other modules (which may contain syntax errors) during simple imports of this module.
slogger = logging  # type: ignore
_SLOGGER_AVAILABLE = False


def _ensure_slogger() -> None:
    """Attempt a lazy import of aidev.logger and set availability flag.

    This is intentionally best-effort and will silently keep using the
    stdlib logging module if the package import fails. Doing the import
    lazily prevents package-level import from triggering unrelated module
    parses at import time.
    """
    global slogger, _SLOGGER_AVAILABLE
    if _SLOGGER_AVAILABLE:
        return
    try:
        # Local import to avoid importing at module import time above
        from .. import logger as _imported_slogger  # type: ignore

        slogger = _imported_slogger  # type: ignore
        _SLOGGER_AVAILABLE = True
    except Exception:
        # Keep default stdlib logging; mark as unavailable
        slogger = logging  # type: ignore
        _SLOGGER_AVAILABLE = False

# HTTP header name expected from the UI when a project is selected. Other
# modules (server, middleware) should import this constant to keep the
# string consistent across the codebase.
# HTTP header expected from UI; HTTP/workspace layers should forward this
# header value into the repo_root argument when invoking apply flows.
HEADER_NAME = "X-AIDEV-PROJECT"

# Explicit exports for other modules that may import these symbols.
__all__ = ["HEADER_NAME", "apply_edits_and_refresh", "apply_and_refresh"]

# Note: perform potentially heavy/packaged imports lazily inside the function
# to avoid importing the whole aidev package at module import time. This
# prevents import-time failures in other modules from breaking simple
# operations that only need this function at runtime.

# Module note: this module exposes two entrypoints. The historical
# apply_edits_and_refresh implements the transactional apply + refresh
# behavior. A compatibility wrapper apply_and_refresh is provided to
# accept a variety of common argument names/positions used across the
# codebase and tests; callers should prefer apply_and_refresh for new
# integrations.

ProgressFn = Callable[[str, Dict[str, Any]], None]
ErrorFn = Callable[[str, Dict[str, Any]], None]


def _ensure_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort; callers handle errors
        logging.debug("Failed to ensure directory for %s", path, exc_info=True)


def _safe_structured_info(event: str, meta: Optional[Dict[str, Any]] = None, fallback_msg: Optional[str] = None) -> None:
    """Emit a structured info log using the structured logger when available.

    If the structured logger isn't available or doesn't accept the meta param,
    fall back to a readable stdlib logging.info call.
    """
    try:
        # Try lazy import of structured logger to avoid package-level imports at module import time
        _ensure_slogger()
        if _SLOGGER_AVAILABLE:
            try:
                # Preferred structured API: info(name, meta={...})
                slogger.info(event, meta=meta)  # type: ignore[arg-type]
                return
            except TypeError:
                # Try alternate common signature: info(msg, extra=...)
                try:
                    slogger.info(event, extra={"meta": meta})  # type: ignore[arg-type]
                    return
                except Exception:
                    pass
        # Fallback to stdlib logging formatting for human visibility
        if meta:
            logging.info("%s %s", fallback_msg or event, json.dumps(meta, ensure_ascii=False))
        else:
            logging.info(fallback_msg or event)
    except Exception:
        logging.debug("Failed to emit structured info log", exc_info=True)


def _safe_structured_error(event: str, meta: Optional[Dict[str, Any]] = None, fallback_msg: Optional[str] = None) -> None:
    """Emit a structured error log similar to _safe_structured_info.

    Preserve as much metadata as possible; never swallow errors.
    """
    try:
        _ensure_slogger()
        if _SLOGGER_AVAILABLE:
            try:
                slogger.error(event, meta=meta)  # type: ignore[arg-type]
                return
            except TypeError:
                try:
                    slogger.error(event, extra={"meta": meta})  # type: ignore[arg-type]
                    return
                except Exception:
                    pass
        # Fallback
        if meta:
            logging.error("%s %s", fallback_msg or event, json.dumps(meta, ensure_ascii=False))
        else:
            logging.error(fallback_msg or event)
    except Exception:
        logging.debug("Failed to emit structured error log", exc_info=True)


def apply_edits_and_refresh(
    *,
    root: Path,
    repo_root: Optional[Path] = None,
    kb: Any,
    meta: Dict[str, Any],
    st: Any,
    rec: Dict[str, Any],
    rec_edits: List[Dict[str, Any]],
    stats_obj: Any,
    progress_cb: ProgressFn,
    progress_error_cb: ErrorFn,
    job_id: Optional[str] = None,
    writes_by_rec: Optional[Dict[str, List[str]]] = None,
    rec_id: Optional[str] = None,
    selected: bool = True,
    dry_run: bool = False,
) -> Any:
    """
    Apply a single recommendation's edits transactionally, then refresh
    cards and project_map so subsequent recommendations see updated state.

    When dry_run is False (default) behavior is unchanged: edits are applied
    to `root` and the function returns a list of changed file paths.

    When dry_run is True the edits are applied to a temporary copy of the
    workspace; no file under `root` is modified. Instead the function
    returns a dict containing a JSONL patch (string), a structured
    pre-apply report (format/lint/test results), and a list of changed
    paths discovered in the temp workspace. This enables a deterministic
    pre-apply / dry-run flow that tools can present to users.

    Returns:
        If dry_run is False: List[str] of changed paths (same as before).
        If dry_run is True: Dict with keys:
            - jsonl_patch: str (one JSON record per edit)
            - preapply_report: Dict[str, Any]
            - changed_paths: List[str]

    Raises:
        Exception: if the transactional apply fails. Callers are expected
        to catch this and convert it into a structured ApplyRecResult.
    """
    # Accept repo_root for compatibility; prefer repo_root if provided.
    if repo_root is not None:
        root = repo_root

    # Normalize root to a Path in case callers pass a string
    try:
        root = Path(root)
    except Exception:
        # Let errors surface normally below if root is invalid
        pass

    # Local imports to avoid package-level import-time issues
    try:
        from ..cards import KnowledgeBase  # type: ignore
    except Exception:
        KnowledgeBase = None  # type: ignore

    try:
        from ..io_utils import apply_edits_transactionally  # type: ignore
    except Exception:
        # Provide a clear fallback that will raise at call-time if the util is
        # genuinely unavailable rather than failing at import-time.
        def apply_edits_transactionally(*_args, **_kwargs):  # type: ignore
            raise RuntimeError("apply_edits_transactionally is not available")

    try:
        from .. import runtime as _runtime  # type: ignore
    except Exception:
        _runtime = None  # type: ignore

    # v2 project_map builder (preferred); fallback to KnowledgeBase.save_project_map.
    try:
        from ..repo_map import build_project_map as _build_project_map  # type: ignore
    except Exception:
        _build_project_map = None  # type: ignore

    # Use explicit rec_id param when provided; otherwise fall back to rec dict.
    rid = str(rec_id or rec.get("id") or "rec-unknown")
    t0_apply = time.time()

    # Basic visibility into what we're about to do.
    incoming_paths = sorted(
        {str(e.get("path") or "").strip() for e in rec_edits if e.get("path")}
    )

    # Structured start log (non-blocking); fall back to the original info string.
    try:
        _safe_structured_info(
            "apply.start",
            meta={
                "rec_id": rid,
                "job_id": job_id,
                "edits": len(rec_edits),
                "incoming_paths": incoming_paths,
                "dry_run": bool(dry_run),
            },
            fallback_msg=(
                "[apply_and_refresh] start rec_id=%s job_id=%s edits=%d paths=%s dry_run=%s"
                % (rid, job_id, len(rec_edits), incoming_paths, dry_run)
            ),
        )
    except Exception:
        # ensure logging doesn't break behavior
        try:
            logging.info(
                "[apply_and_refresh] start rec_id=%s job_id=%s edits=%d paths=%s dry_run=%s",
                rid,
                job_id,
                len(rec_edits),
                incoming_paths,
                dry_run,
            )
        except Exception:
            pass

    # Attempt to resolve and record the model that will be used for this stage
    # This makes apply flows explicit about which LLM model is preferred for
    # the 'rec_apply' stage and records it via the llm_client logging API if
    # available. This is resilient to missing modules and does not change
    # apply behavior when unavailable.
    try:
        _llm_client = None
        _cfg = None
        try:
            from .. import llm_client as _llm_client  # type: ignore
        except Exception:
            _llm_client = None
        try:
            from .. import config as _cfg  # type: ignore
        except Exception:
            _cfg = None

        resolved_model = None
        try:
            if _cfg is not None and hasattr(_cfg, "get_model_for"):
                try:
                    resolved_model = _cfg.get_model_for("rec_apply")
                except Exception:
                    resolved_model = None
            # fallback to alternate name if present
            if resolved_model is None and _cfg is not None and hasattr(_cfg, "get_model_for_stage"):
                try:
                    resolved_model = _cfg.get_model_for_stage("rec_apply")
                except Exception:
                    resolved_model = None
        except Exception:
            resolved_model = None

        # Try to record the resolved model via llm_client. Prefer the public log_llm_call API,
        # but fall back to an internal _emit_llm_log shim if present. As a last resort we use
        # _safe_structured_info. This is intentionally tolerant because the structured logging shim
        # may or may not be available at runtime.
        try:
            if _llm_client is not None:
                # Prefer public API
                log_fn = getattr(_llm_client, "log_llm_call", None)
                if callable(log_fn):
                    try:
                        log_fn(stage="rec_apply", model=resolved_model, meta={"rec_id": rid, "job_id": job_id})
                    except TypeError:
                        try:
                            log_fn("rec_apply", model=resolved_model, meta={"rec_id": rid, "job_id": job_id})
                        except TypeError:
                            try:
                                log_fn({"stage": "rec_apply", "resolved_model": resolved_model, "rec_id": rid, "job_id": job_id})
                            except Exception:
                                # Last resort structured info
                                _safe_structured_info(
                                    "llm.resolved",
                                    meta={"stage": "rec_apply", "model": resolved_model, "rec_id": rid, "job_id": job_id},
                                    fallback_msg=("[apply_and_refresh] resolved LLM model for rec_apply: %s" % resolved_model),
                                )
                else:
                    # Try internal shim if public API not present
                    emit_fn = getattr(_llm_client, "_emit_llm_log", None)
                    if callable(emit_fn):
                        try:
                            emit_fn("rec_apply", model=resolved_model, meta={"rec_id": rid, "job_id": job_id})
                        except TypeError:
                            try:
                                emit_fn(stage="rec_apply", model=resolved_model, meta={"rec_id": rid, "job_id": job_id})
                            except Exception:
                                _safe_structured_info(
                                    "llm.resolved",
                                    meta={"stage": "rec_apply", "model": resolved_model, "rec_id": rid, "job_id": job_id},
                                    fallback_msg=("[apply_and_refresh] resolved LLM model for rec_apply: %s" % resolved_model),
                                )
                    else:
                        _safe_structured_info(
                            "llm.resolved",
                            meta={"stage": "rec_apply", "model": resolved_model, "rec_id": rid, "job_id": job_id},
                            fallback_msg=("[apply_and_refresh] resolved LLM model for rec_apply: %s" % resolved_model),
                        )
            else:
                _safe_structured_info(
                    "llm.resolved",
                    meta={"stage": "rec_apply", "model": resolved_model, "rec_id": rid, "job_id": job_id},
                    fallback_msg=("[apply_and_refresh] resolved LLM model for rec_apply: %s" % resolved_model),
                )
        except Exception:
            _safe_structured_info(
                "llm.resolved",
                meta={"stage": "rec_apply", "model": resolved_model, "rec_id": rid, "job_id": job_id},
                fallback_msg=("[apply_and_refresh] resolved LLM model for rec_apply: %s" % resolved_model),
            )
    except Exception:
        # Do not allow model resolution/logging to interrupt apply flow
        logging.debug("Failed to resolve or log llm model for rec_apply", exc_info=True)

    # Validate repo root early: callers (HTTP layer/orchestrator) rely on a clear
    # validation error so they can return a 400/422-level response. This ensures
    # we never silently operate against a non-existent or non-directory root.
    try:
        root = Path(root)
    except Exception:
        # If we can't coerce to Path, emit via progress_error_cb and raise
        msg = f"Provided repo_root/root '{root}' is not a valid path"
        try:
            progress_error_cb("invalid_repo_root", {"error": msg, "rec_id": rid, "job_id": job_id})
        except Exception:
            logging.debug("Failed to emit invalid_repo_root via progress_error_cb", exc_info=True)
        raise ValueError(msg)

    if not root.exists() or not root.is_dir():
        msg = f"Provided repo_root/root '{root}' does not exist or is not a directory"
        try:
            progress_error_cb("invalid_repo_root", {"error": msg, "rec_id": rid, "job_id": job_id})
        except Exception:
            logging.debug("Failed to emit invalid_repo_root via progress_error_cb", exc_info=True)
        raise ValueError(msg)

    # Compute a canonical, resolved project_root that will be passed to lower-level writers
    try:
        project_root_canonical = Path(root).resolve()
    except Exception:
        # Fallback to a Path object even if resolve fails for some reason
        project_root_canonical = Path(root)

    # Small debug/log to make the coupling explicit: we resolve and pass this
    # Path to io_utils.apply_edits_transactionally below. TODO: io_utils.apply_edits_transactionally
    # must enforce the path-safety contract that all writes are constrained to
    # the provided repo_root. See follow-up requirement in repository notes.
    try:
        logging.debug(
            "apply_and_refresh operating on repo_root=%s (resolved=%s)", root, project_root_canonical
        )
    except Exception:
        # Best-effort logging; ignore failures
        pass

    # ---- dry-run path: apply into a temp copy, run sandboxed checks ----
    if dry_run:
        try:
            with tempfile.TemporaryDirectory(prefix="aidev-dryrun-") as td:
                tmp_root = Path(td) / "workspace"
                # Copy project into temp workspace; ignore errors but try to preserve tree
                try:
                    shutil.copytree(root, tmp_root)
                except Exception:
                    # If root is empty or copy fails, ensure tmp_root exists
                    tmp_root.mkdir(parents=True, exist_ok=True)

                # Apply edits into the temp workspace (writes only to temp copy)
                try:
                    # pass project_root so io_utils can enforce path-safety / root-locked writes
                    apply_edits_transactionally(
                        root=tmp_root, project_root=project_root_canonical, edits=rec_edits, dry_run=False, stats=stats_obj, st=st
                    )
                except Exception as e:
                    progress_error_cb(
                        "preapply_apply_failed",
                        {
                            "error": str(e),
                            "trace": traceback.format_exc(),
                            "rec_id": rid,
                            "job_id": job_id,
                        },
                    )
                    # Structured error emit for visibility (non-fatal to control flow beyond re-raise)
                    _safe_structured_error(
                        "preapply_apply_failed",
                        meta={"error": str(e), "trace": traceback.format_exc(), "rec_id": rid, "job_id": job_id},
                        fallback_msg=(
                            "[apply_and_refresh] preapply apply failed rec_id=%s job_id=%s: %s" % (rid, job_id, e)
                        ),
                    )
                    raise

                # Collect changed paths relative to project root inside temp copy
                changed_paths = sorted({e.get("path") for e in rec_edits if e.get("path")})

                # Build JSONL patch output: one JSON entry per edit
                try:
                    jsonl_patch = "\n".join(
                        json.dumps(e, ensure_ascii=False) for e in rec_edits
                    )
                except Exception:
                    jsonl_patch = json.dumps(rec_edits, ensure_ascii=False)

                # ---- run lightweight pre-apply checks ----
                preapply_report: Dict[str, Any] = {
                    "formatters": {},
                    "linters": {},
                    "tests": {},
                }

                # Formatter check: for Python files, use a lightweight check but avoid
                # instantiating runtime-based formatters here to prevent any
                # potential LLM/IO side-effects during dry-run. Mark as skipped
                # when we cannot run a local formatter.
                try:
                    fmt_issues: List[Dict[str, Any]] = []
                    py_formatter = None

                    for e in rec_edits:
                        p = e.get("path")
                        if not p:
                            continue
                        if Path(p).suffix.lower() != ".py":
                            continue
                        target = tmp_root / p
                        if not target.exists():
                            continue
                        try:
                            text = target.read_text(encoding="utf-8")
                        except Exception:
                            continue
                        if py_formatter is None:
                            # can't run formatter in this environment; mark as skipped
                            fmt_issues.append({"path": p, "skipped": "no_formatter"})
                            continue
                        try:
                            formatted = py_formatter.format_file(target, text)  # type: ignore[arg-type]
                        except Exception:
                            # If formatter fails for a file, record an issue for visibility
                            fmt_issues.append({"path": p, "error": "formatter_error"})
                            continue
                        if formatted != text:
                            fmt_issues.append(
                                {"path": p, "before_len": len(text), "after_len": len(formatted)}
                            )
                    preapply_report["formatters"]["python_simple_normalize"] = {
                        "passed": len([i for i in fmt_issues if "error" in i or "before_len" in i]) == 0,
                        "issues": fmt_issues,
                    }
                except Exception:
                    preapply_report["formatters"]["python_simple_normalize"] = {
                        "passed": False,
                        "error": "formatter_check_exception",
                    }

                # Linter check: try running flake8 in temp workspace if available
                try:
                    lint_proc = subprocess.run(
                        ["flake8"],
                        cwd=str(tmp_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=30,
                    )
                    lint_out = lint_proc.stdout or ""
                    preapply_report["linters"]["flake8"] = {
                        "rc": lint_proc.returncode,
                        "output": lint_out,
                        "passed": lint_proc.returncode == 0,
                    }
                except FileNotFoundError:
                    preapply_report["linters"]["flake8"] = {
                        "skipped": True,
                        "reason": "flake8-not-installed",
                    }
                except subprocess.TimeoutExpired:
                    preapply_report["linters"]["flake8"] = {"skipped": False, "error": "timeout"}
                except Exception as e:
                    preapply_report["linters"]["flake8"] = {"error": str(e)}

                # Test check: try running pytest if present
                try:
                    test_proc = subprocess.run(
                        ["pytest", "-q"],
                        cwd=str(tmp_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=60,
                    )
                    test_out = test_proc.stdout or ""
                    preapply_report["tests"]["pytest"] = {
                        "rc": test_proc.returncode,
                        "output": test_out,
                        "passed": test_proc.returncode == 0,
                    }
                except FileNotFoundError:
                    preapply_report["tests"]["pytest"] = {
                        "skipped": True,
                        "reason": "pytest-not-installed",
                    }
                except subprocess.TimeoutExpired:
                    preapply_report["tests"]["pytest"] = {"skipped": False, "error": "timeout"}
                except Exception as e:
                    preapply_report["tests"]["pytest"] = {"error": str(e)}

                # Summarize decision: blocked if any test or linter failed (rc != 0)
                blocked_reasons: List[str] = []
                try:
                    lf = preapply_report.get("linters", {}).get("flake8", {})
                    if lf and not lf.get("skipped", False) and lf.get("rc") is not None:
                        if lf.get("rc") != 0:
                            blocked_reasons.append("lint_fail")
                except Exception:
                    pass
                try:
                    tf = preapply_report.get("tests", {}).get("pytest", {})
                    if tf and not tf.get("skipped", False) and tf.get("rc") is not None:
                        if tf.get("rc") != 0:
                            blocked_reasons.append("tests_fail")
                except Exception:
                    pass

                preapply_report["decision"] = "blocked" if blocked_reasons else "preapply"
                preapply_report["blocked_reasons"] = blocked_reasons

                # Emit a progress event for UI / orchestrator
                try:
                    progress_cb(
                        "preapply_result",
                        {
                            "rec_id": rid,
                            "job_id": job_id,
                            "changed_paths": changed_paths,
                            "jsonl_patch": jsonl_patch,
                            "preapply_report": preapply_report,
                        },
                    )
                except Exception:
                    logging.debug("Failed to emit preapply_result event", exc_info=True)

                # Return structured dry-run result
                return {
                    "jsonl_patch": jsonl_patch,
                    "preapply_report": preapply_report,
                    "changed_paths": changed_paths,
                }
        except Exception as e:
            progress_error_cb(
                "preapply_failed",
                {
                    "error": str(e),
                    "trace": traceback.format_exc(),
                    "rec_id": rid,
                    "job_id": job_id,
                },
            )
            _safe_structured_error(
                "preapply_failed",
                meta={"error": str(e), "trace": traceback.format_exc(), "rec_id": rid, "job_id": job_id},
                fallback_msg=(
                    "[apply_and_refresh] preapply dry-run failed rec_id=%s job_id=%s: %s" % (rid, job_id, e)
                ),
            )
            logging.error(
                "[apply_and_refresh] preapply dry-run failed rec_id=%s job_id=%s: %s",
                rid,
                job_id,
                e,
            )
            raise

    # ---- transactional apply (existing behavior) ----
    try:
        # pass project_root so io_utils can enforce path-safety / root-locked writes
        # Log the resolved root being passed to the IO layer so callers and
        # operators can trace which repository was targeted for writes.
        try:
            logging.debug(
                "Calling apply_edits_transactionally with root=%s (resolved=%s)", root, project_root_canonical
            )
        except Exception:
            pass
        # TODO: io_utils.apply_edits_transactionally must validate and enforce
        # that all writes are constrained to the provided repo_root (path-safety contract).
        apply_edits_transactionally(root=root, project_root=project_root_canonical, edits=rec_edits, dry_run=False, stats=stats_obj, st=st)
    except Exception as e:
        progress_error_cb(
            "apply_edits",
            {
                "error": str(e),
                "trace": traceback.format_exc(),
                "rec_id": rid,
                "job_id": job_id,
            },
        )
        # Structured error (non-blocking) with meta for easier parsing
        _safe_structured_error(
            "apply_edits_failed",
            meta={"error": str(e), "trace": traceback.format_exc(), "rec_id": rid, "job_id": job_id},
            fallback_msg=(
                "[apply_and_refresh] apply_edits_transactionally failed rec_id=%s job_id=%s: %s" % (rid, job_id, e)
            ),
        )
        logging.error(
            "[apply_and_refresh] apply_edits_transactionally failed "
            "rec_id=%s job_id=%s: %s",
            rid,
            job_id,
            e,
        )
        # Surface the failure to the caller so it can set ApplyRecResult.reason
        raise

    elapsed = round(time.time() - t0_apply, 3)
    try:
        st.trace.write(
            "APPLY",
            "transaction",
            {
                "rec_id": rid,
                "count": len(rec_edits),
                "elapsed_sec": elapsed,
            },
        )
    except Exception:
        logging.debug("Failed to write APPLY transaction trace", exc_info=True)

    # ---- collect affected paths from the edits ----
    paths = sorted({e.get("path") for e in rec_edits if e.get("path")})

    # Record writes_by_rec for caller if provided
    if paths and writes_by_rec is not None:
        writes_by_rec.setdefault(rid, []).extend(paths)

    # Ensure we only report paths that live under the project root.
    root_resolved = project_root_canonical
    filtered_paths: List[str] = []
    for p in paths:
        try:
            candidate = (root / p).resolve()
            # Use relative_to to ensure candidate is within root
            candidate.relative_to(root_resolved)
            # path is inside project root
            filtered_paths.append(p)
        except Exception:
            # Skip paths outside project root but log an error for visibility
            progress_error_cb(
                "apply_out_of_root_path",
                {
                    "error": "Path outside project root",
                    "path": p,
                    "rec_id": rid,
                    "job_id": job_id,
                },
            )
            _safe_structured_error(
                "apply_out_of_root_path",
                meta={"error": "Path outside project root", "path": p, "rec_id": rid, "job_id": job_id},
                fallback_msg=(
                    "[apply_and_refresh] Applied path appears outside project root rec_id=%s job_id=%s path=%s" % (rid, job_id, p)
                ),
            )

    # Compute created/modified/skipped heuristically from rec_edits for structured summaries
    try:
        created_count = sum(1 for e in rec_edits if e.get("is_new"))
        modified_count = sum(1 for e in rec_edits if e.get("path") and not e.get("is_new"))
        skipped_count = max(0, len(rec_edits) - created_count - modified_count)
    except Exception:
        created_count = 0
        modified_count = 0
        skipped_count = 0

    # Structured applied summary (emit before writing trace/app.log)
    try:
        _safe_structured_info(
            "apply.applied",
            meta={
                "rec_id": rid,
                "job_id": job_id,
                "num_files": len(filtered_paths),
                "changed_paths": filtered_paths,
                "elapsed_sec": elapsed,
                "dry_run": bool(dry_run),
                "created": created_count,
                "modified": modified_count,
                "skipped": skipped_count,
            },
            fallback_msg=(
                "[apply_and_refresh] applied rec_id=%s job_id=%s num_files=%d paths=%s elapsed=%.3fs" % (
                    rid, job_id, len(filtered_paths), filtered_paths, elapsed
                )
            ),
        )
    except Exception:
        logging.debug("Failed to emit structured apply.applied log", exc_info=True)

    logging.info(
        "[apply_and_refresh] applied rec_id=%s job_id=%s num_files=%d paths=%s "
        "elapsed=%.3fs",
        rid,
        job_id,
        len(filtered_paths),
        filtered_paths,
        elapsed,
    )

    # Only write trace/app.log entries when there are applied changes to record.
    if filtered_paths:
        # compute safe project_root string for traces/events; avoid raising here
        try:
            project_root_str = str(project_root_canonical)
        except Exception:
            project_root_str = str(root)

        # ---- append JSONL trace record ----
        try:
            trace_path = root / ".aidev" / "trace.jsonl"
            _ensure_dir(trace_path)
            trace_record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "rec_id": rid,
                "recommendation_id": rid,
                "selected": bool(selected),
                "changed_paths": filtered_paths,
                "project_root": project_root_str,
            }
            with trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
        except Exception:
            logging.debug("Failed to append trace.jsonl record", exc_info=True)
            # non-fatal: report but continue
            progress_error_cb(
                "trace_append_failed",
                {
                    "error": "Failed to append .aidev/trace.jsonl",
                    "rec_id": rid,
                    "job_id": job_id,
                },
            )

        # ---- write a human-readable app.log entry ----
        try:
            app_log = root / ".aidev" / "app.log"
            _ensure_dir(app_log)
            ts = datetime.utcnow().isoformat() + "Z"
            try:
                pr = project_root_str
            except Exception:
                try:
                    pr = str(root)
                except Exception:
                    pr = "<unknown>"
            with app_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"{ts} Applied changes: rec_id={rid} recommendation_id={rid} "
                    f"selected={selected} project_root={pr} paths={filtered_paths}\n"
                )
        except Exception:
            logging.debug("Failed to write app.log entry", exc_info=True)
            progress_error_cb(
                "app_log_failed",
                {
                    "error": "Failed to write .aidev/app.log",
                    "rec_id": rid,
                    "job_id": job_id,
                },
            )

    # ---- runtime events for UI / orchestrator ----
    try:
        # ensure we include project_root in emitted payloads for auditing/UI
        try:
            project_root_str = str(project_root_canonical)
        except Exception:
            project_root_str = str(root)

        progress_cb(
            "applied_changes",
            {
                "rec_id": rid,
                "num_files": len(filtered_paths),
                "changed_paths": filtered_paths,
                "job_id": job_id,
                "project_root": project_root_str,
            },
        )
    except Exception:
        logging.debug("Failed to emit applied_changes event", exc_info=True)

    try:
        # include project_root in final apply_done event as well
        try:
            project_root_str = str(project_root_canonical)
        except Exception:
            project_root_str = str(root)

        progress_cb(
            "apply_done",
            {
                "rec_id": rid,
                "count": len(rec_edits),
                "elapsed_sec": elapsed,
                "job_id": job_id,
                "project_root": project_root_str,
            },
        )
    except Exception:
        logging.debug("Failed to emit apply_done event", exc_info=True)

    # ---- small helpers for atomic writes and index building ----
    def _atomic_write_json(out_path: Path, data: Any) -> None:
        """
        Atomically write JSON to out_path using a same-directory temp file.
        """
        _ensure_dir(out_path)
        out_dir = out_path.parent
        tmp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(out_dir),
                delete=False,
                mode="w",
                encoding="utf-8",
            ) as tf:
                tmp_name = tf.name
                json.dump(data, tf, ensure_ascii=False, indent=2)
                tf.flush()
                os.fsync(tf.fileno())
            # Atomic replace
            os.replace(tmp_name, str(out_path))
        finally:
            # Best-effort cleanup if a temp file is left behind
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except Exception:
                    logging.debug(
                        "Failed to remove temp file %s during atomic write",
                        tmp_name,
                        exc_info=True,
                    )

    def _build_cards_index(fpaths: List[str]) -> List[Dict[str, Any]]:
        """
        Build a simple index of source files + card metadata, including checksums.
        """
        out_index: List[Dict[str, Any]] = []
        seen_paths = set()

        # Include entries for changed project files (path + checksum of source file)
        for p in fpaths:
            try:
                fp = root / p
                if fp.exists():
                    try:
                        data_bytes = fp.read_bytes()
                        chksum = hashlib.sha256(data_bytes).hexdigest()
                    except Exception:
                        chksum = None
                    out_index.append({"path": p, "checksum": chksum})
                    seen_paths.add(p)
            except Exception:
                logging.debug("Failed to stat/checksum path %s", p, exc_info=True)

        # Scan .aidev/cards/*.card.json for additional card metadata
        try:
            cards_glob = root / ".aidev" / "cards" / "*.card.json"
            for card_file in glob.glob(str(cards_glob)):
                try:
                    cf_path = Path(card_file)
                    text = cf_path.read_text(encoding="utf-8")
                    obj = json.loads(text)
                    card_path: Optional[str] = None

                    # Try several common keys that might reference a source path
                    for k in ("path", "source", "file", "filepath"):
                        if isinstance(obj.get(k), str):
                            card_path = obj.get(k)
                            break
                    if not card_path:
                        # Fallback to using the card filename as an identifier
                        card_path = str(cf_path.name)

                    if card_path in seen_paths:
                        continue

                    try:
                        csum = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    except Exception:
                        csum = None

                    out_index.append(
                        {
                            "path": card_path,
                            "card_file": str(cf_path.relative_to(root)),
                            "checksum": csum,
                            "summary": obj.get("summary"),
                        }
                    )
                    seen_paths.add(card_path)
                except Exception:
                    logging.debug("Failed to read card file %s", card_file, exc_info=True)
        except Exception:
            logging.debug("Failed scanning .aidev/cards for index", exc_info=True)

        return out_index

    def _try_kb_write_cards_index(cards_index_path: Path) -> bool:
        """
        Try a few plausible KnowledgeBase APIs to write the cards index.

        Returns True if a KB method appears to succeed, False otherwise.
        """
        if kb is None:
            return False

        candidates = [
            "write_cards_index",
            "save_cards_index",
            "write_index",
            "write_cards_index_file",
        ]
        for name in candidates:
            fn = getattr(kb, name, None)
            if not callable(fn):
                continue
            try:
                try:
                    fn(cards_index_path)
                    return True
                except TypeError:
                    # Some variants may take no args
                    fn()
                    return True
            except Exception:
                logging.debug("KB API %s failed", name, exc_info=True)
        return False

    # Small metadata fields default
    meta.setdefault("cards_index_written", False)
    meta.setdefault("project_map_written", False)

    # ---- refresh card index incrementally + project_map (v2) ----
    try:
        # Ensure we have a KnowledgeBase instance; reuse provided kb when available.
        if kb is None:  # type: ignore[truthy-function]
            structure = getattr(st, "structure", None) or {}
            if "KnowledgeBase" in globals() and KnowledgeBase is not None:  # type: ignore[name-defined]
                try:
                    kb = KnowledgeBase(root=root, structure=structure)  # type: ignore[assignment]
                except Exception:
                    # Fallback to an empty structure if the provided one fails
                    kb = KnowledgeBase(root=root, structure={})  # type: ignore[assignment]
            else:
                kb = None

        # Small runtime flag to prefer the heuristic-only, non-LLM refresh path.
        prefer_heuristic = True
        if _runtime is not None:
            try:
                prefer_heuristic = getattr(_runtime, "PREFER_HEURISTIC_REFRESH", True)
            except Exception:
                prefer_heuristic = True

        pm: Optional[Dict[str, Any]] = None

        # Per-recommendation refresh guard state: if a specialized KB helper runs
        # successfully we will skip the older kb.update_cards call to avoid double-work.
        refresh_state: Dict[str, Any] = {"skipped_kb_update": False}

        
        # Lazy-read config flag REFRESH_CARDS_BETWEEN_RECS (default True)
        try:
            try:
                from .. import config as _local_cfg  # type: ignore
                refresh_between = getattr(_local_cfg, "REFRESH_CARDS_BETWEEN_RECS", True)
            except Exception:
                refresh_between = True
        except Exception:
            refresh_between = True

        # Helper to attempt the focused per-recommendation refresh using a
        # KB-provided helper when available. This emits progress events and
        # records meta keys when a refresh actually ran. Errors are reported
        # via progress_error_cb and do not abort the overall apply flow.
        def _maybe_run_per_rec_refresh() -> Optional[List[str]]:
            # Only run when feature is enabled and there are changed files
            if not refresh_between:
                return None
            if not filtered_paths:
                return None
            if kb is None:
                return None

            # Prefer aidev.events API for consistent SSE/trace formatting; fallback to progress_cb
            _events = None
            try:
                from .. import events as _events  # type: ignore
            except Exception:
                _events = None

            # Emit start event via events helper when present, else progress_cb
            
            try:
                payload_start = {"changed_paths": filtered_paths, "rec_id": rid}
                if _events is not None and hasattr(_events, "cards_refresh_start"):
                    try:
                        _events.cards_refresh_start(payload_start)
                    except Exception:
                        logging.debug("events.cards_refresh_start failed", exc_info=True)
                else:
                    try:
                        progress_cb("cards.refresh.start", payload_start)
                    except Exception:
                        logging.debug("Failed to emit cards.refresh.start via progress_cb", exc_info=True)
            except Exception:
                # never fail the refresh start emission path
                logging.debug("Failed preparing cards refresh start emission", exc_info=True)

            refreshed_paths: Optional[List[str]] = None
            try:
                fn = getattr(kb, "refresh_changed_paths", None)
                if callable(fn):
                    try:
                        # Try positional then kw-arg signature
                        
                        try:
                            res = fn(filtered_paths)
                        except TypeError:
                            res = fn(files=filtered_paths)  # type: ignore[arg-type]

                        # Interpret result shapes
                        if isinstance(res, list):
                            refreshed_paths = list(res)
                        elif isinstance(res, dict):
                            for key in ("refreshed_paths", "paths", "changed_paths", "updated_paths", "files"):
                                if isinstance(res.get(key), list):
                                    refreshed_paths = list(res.get(key))
                                    break
                        # If we could not interpret the result, treat as failure and fall back
                        if refreshed_paths is None:
                            raise ValueError("refresh_changed_paths returned unexpected shape")
                        # Mark that we successfully refreshed via helper and can skip older update path
                        refresh_state["skipped_kb_update"] = True
                    except Exception:
                        # Helper failed; fall back to older update_cards below
                        logging.debug("KB.refresh_changed_paths failed; will fall back to kb.update_cards", exc_info=True)
                        refreshed_paths = None
                else:
                    refreshed_paths = None

                # If helper did not run or failed, fallback to update_cards so behavior remains unchanged
                if refreshed_paths is None:
                    try:
                        # Reuse existing update_cards signature behavior; preserve try/except semantics of callers
                        if filtered_paths:
                            try:
                                kb.update_cards(files=filtered_paths, changed_only=True)  # type: ignore[attr-defined]
                            except TypeError:
                                kb.update_cards(changed_only=True)  # type: ignore[attr-defined]
                        else:
                            kb.update_cards(changed_only=True)  # type: ignore[attr-defined]
                        # Assume updated for metrics when fallback succeeds
                        refreshed_paths = filtered_paths
                    except Exception as e:
                        # Report but do not raise
                        progress_error_cb(
                            "kb_update_failed",
                            {
                                "error": str(e),
                                "trace": traceback.format_exc(),
                                "rec_id": rid,
                                "job_id": job_id,
                            },
                        )
                        refreshed_paths = []

            except Exception as e:
                # Catch-all for unexpected issues; report and continue
                try:
                    progress_error_cb(
                        "cards_refresh_failed",
                        {
                            "error": str(e),
                            "trace": traceback.format_exc(),
                            "rec_id": rid,
                            "job_id": job_id,
                        },
                    )
                except Exception:
                    logging.debug("Failed to emit cards_refresh_failed", exc_info=True)
                refreshed_paths = []

            # Canonicalize refreshed_paths to a list (possibly empty)
            try:
                if refreshed_paths is None:
                    refreshed_paths = []
                else:
                    # ensure strings and dedupe while preserving order
                    seen = set()
                    ordered: List[str] = []
                    for x in refreshed_paths:
                        if not isinstance(x, str):
                            continue
                        if x in seen:
                            continue
                        seen.add(x)
                        ordered.append(x)
                    refreshed_paths = ordered
            except Exception:
                refreshed_paths = []

            # Record metadata and emit done event via events helper when present
            try:
                meta["cards_refreshed_paths"] = refreshed_paths
                meta["cards_refreshed_count"] = len(refreshed_paths)
                meta["cards_refreshed_ts"] = datetime.utcnow().isoformat() + "Z"
            except Exception:
                logging.debug("Failed to record cards refresh metadata", exc_info=True)

            try:
                payload_done = {"changed_paths": filtered_paths, "refreshed_count": len(refreshed_paths), "rec_id": rid}
                if _events is not None and hasattr(_events, "cards_refresh_done"):
                    try:
                        _events.cards_refresh_done(payload_done)
                    except Exception:
                        logging.debug("events.cards_refresh_done failed", exc_info=True)
                else:
                    try:
                        progress_cb("cards.refresh.done", payload_done)
                    except Exception:
                        logging.debug("Failed to emit cards.refresh.done via progress_cb", exc_info=True)
            except Exception:
                logging.debug("Failed to emit cards.refresh.done", exc_info=True)

            return refreshed_paths

        # Attempt a per-recommendation refresh now (this runs before project_map refresh)
        
        try:
            # Only attempt when we have a kb instance
            _maybe_run_per_rec_refresh()
        except Exception:
            # Guard against any accidental exceptions from the helper logic
            logging.debug("Per-recommendation refresh raised unexpectedly", exc_info=True)

        # ---- existing heuristic vs full refresh logic follows ----
        if prefer_heuristic:
            # Heuristic-only update: avoid any LLM enrichment. Call update_cards in
            # heuristic / changed-only mode so work is deterministic and cheap.
            if kb is not None and not refresh_state.get("skipped_kb_update", False):
                try:
                    if filtered_paths:
                        try:
                            kb.update_cards(files=filtered_paths, changed_only=True)  # type: ignore[attr-defined]
                        except TypeError:
                            kb.update_cards(changed_only=True)  # type: ignore[attr-defined]
                    else:
                        kb.update_cards(changed_only=True)  # type: ignore[attr-defined]
                except Exception as e:
                    progress_error_cb(
                        "kb_update_failed",
                        {
                            "error": str(e),
                            "trace": traceback.format_exc(),
                            "rec_id": rid,
                            "job_id": job_id,
                        },
                    )

            # Refresh project_map using repo_map.build_project_map (v2 canonical),
            # falling back to KnowledgeBase.save_project_map only if needed.
            try:
                if _build_project_map is not None:
                    # Prefer to pass refreshed paths to build_project_map if available in meta
                    try:
                        refreshed = meta.get("cards_refreshed_paths")
                        if refreshed:
                            try:
                                pm_any = _build_project_map(root, changed_paths=refreshed, force=False)
                            except TypeError:
                                pm_any = _build_project_map(root, force=False)
                        else:
                            pm_any = _build_project_map(root, force=False)
                    except Exception:
                        # Last-resort: call without changed_paths
                        pm_any = _build_project_map(root, force=False)
                    if isinstance(pm_any, dict):
                        pm = pm_any
                        try:
                            meta["project_map"] = pm_any
                        except Exception:
                            pass
                    pm_path = root / ".aidev" / "project_map.json"
                    try:
                        progress_cb(
                            "project_map_refresh",
                            {
                                "path": str(pm_path),
                                "rec_id": rid,
                                "job_id": job_id,
                                "mode": "heuristic_only",
                                "total_files": pm_any.get("total_files")
                                if isinstance(pm_any, dict)
                                else None,
                            },
                        )
                    except Exception:
                        logging.debug("Failed to emit project_map_refresh event", exc_info=True)
                else:
                    # Legacy fallback: use KnowledgeBase.save_project_map if available.
                    out_path = root / ".aidev" / "project_map.json"
                    _ensure_dir(out_path)

                    save_fn = getattr(kb, "save_project_map", None) if kb is not None else None
                    if callable(save_fn):
                        try:
                            # Attach refreshed paths into project_meta if present so older save functions can include them
                            pm_meta = dict(meta)
                            if meta.get("cards_refreshed_paths"):
                                pm_meta["cards_refreshed_paths"] = meta.get("cards_refreshed_paths")
                            data_path = save_fn(
                                out_path,
                                project_meta=pm_meta,
                                include_tree=False,
                                include_files=True,
                                prefer_ai_summaries=True,
                                compact_tree=True,
                                pretty=False,
                            )
                        except TypeError:
                            # Fallback for older signatures
                            data_path = save_fn(out_path, project_meta=meta)

                        try:
                            progress_cb(
                                "project_map_refresh",
                                {
                                    "path": str(data_path),
                                    "rec_id": rid,
                                    "job_id": job_id,
                                    "mode": "heuristic_only",
                                },
                            )
                        except Exception:
                            logging.debug("Failed to emit project_map_refresh event", exc_info=True)
                    else:
                        raise RuntimeError(
                            "project_map refresh failed: neither repo_map.build_project_map "
                            "nor KnowledgeBase.save_project_map is available"
                        )
            except Exception as e:
                progress_error_cb(
                    "project_map_post_apply",
                    {
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "rec_id": rid,
                        "job_id": job_id,
                    },
                )

            # Emit a clear, standardized trace/event for heuristic refreshes so
            # pipeline traces show that no LLM enrichment was performed.
            try:
                progress_cb(
                    "heuristic_card_refresh",
                    {
                        "rec_id": rid,
                        "job_id": job_id,
                        "changed_paths": filtered_paths,
                        "mode": "heuristic_only",
                    },
                )
            except Exception:
                logging.debug("Failed to emit heuristic_card_refresh event", exc_info=True)

        else:
            # Full path: allow KnowledgeBase to run richer updates (may include AI summaries).
            if kb is not None and not refresh_state.get("skipped_kb_update", False):
                try:
                    if filtered_paths:
                        kb.update_cards(
                            files=filtered_paths, force=True, changed_only=False
                        )  # type: ignore[attr-defined]
                    else:
                        kb.update_cards(force=False, changed_only=True)  # type: ignore[attr-defined]
                except Exception as e:
                    progress_error_cb(
                        "kb.update_cards_post_apply",
                        {
                            "error": str(e),
                            "trace": traceback.format_exc(),
                            "rec_id": rid,
                            "job_id": job_id,
                        },
                    )

            # Use repo_map.build_project_map as the canonical writer for project_map;
            # fall back to KnowledgeBase.save_project_map only when necessary.
            try:
                if _build_project_map is not None:
                    try:
                        refreshed = meta.get("cards_refreshed_paths")
                        if refreshed:
                            try:
                                pm_any = _build_project_map(root, changed_paths=refreshed, force=True)
                            except TypeError:
                                pm_any = _build_project_map(root, force=True)
                        else:
                            pm_any = _build_project_map(root, force=True)
                    except Exception:
                        pm_any = _build_project_map(root, force=True)
                    if isinstance(pm_any, dict):
                        pm = pm_any
                        try:
                            meta["project_map"] = pm_any
                        except Exception:
                            pass
                    pm_path = root / ".aidev" / "project_map.json"
                    try:
                        progress_cb(
                            "project_map_refresh",
                            {
                                "path": str(pm_path),
                                "rec_id": rid,
                                "job_id": job_id,
                                "mode": "full",
                                "total_files": pm_any.get("total_files")
                                if isinstance(pm_any, dict)
                                else None,
                            },
                        )
                    except Exception:
                        logging.debug(
                            "Failed to emit project_map_refresh event (full)", exc_info=True
                        )
                else:
                    save_fn = getattr(kb, "save_project_map", None) if kb is not None else None
                    if callable(save_fn):
                        out_path = root / ".aidev" / "project_map.json"
                        _ensure_dir(out_path)
                        try:
                            pm_meta = dict(meta)
                            if meta.get("cards_refreshed_paths"):
                                pm_meta["cards_refreshed_paths"] = meta.get("cards_refreshed_paths")
                            try:
                                data_path = save_fn(
                                    out_path,
                                    project_meta=pm_meta,
                                    include_tree=False,
                                    include_files=True,
                                    prefer_ai_summaries=True,
                                    compact_tree=True,
                                    pretty=False,
                                )
                            except TypeError:
                                data_path = save_fn(out_path, project_meta=meta)

                        except Exception:
                            data_path = None

                        try:
                            progress_cb(
                                "project_map_refresh",
                                {
                                    "path": str(data_path) if data_path is not None else str(out_path),
                                    "rec_id": rid,
                                    "job_id": job_id,
                                    "mode": "full",
                                },
                            )
                        except Exception:
                            logging.debug(
                                "Failed to emit project_map_refresh event (full)", exc_info=True
                            )
                    else:
                        raise RuntimeError(
                            "project_map refresh failed: neither repo_map.build_project_map "
                            "nor KnowledgeBase.save_project_map is available"
                        )
            except Exception as e:
                progress_error_cb(
                    "project_map_post_apply",
                    {
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "rec_id": rid,
                        "job_id": job_id,
                    },
                )

            try:
                progress_cb(
                    "heuristic_card_refresh",
                    {
                        "rec_id": rid,
                        "job_id": job_id,
                        "changed_paths": filtered_paths,
                        "mode": "full",
                    },
                )
            except Exception:
                logging.debug(
                    "Failed to emit heuristic_card_refresh event (full)", exc_info=True
                )

        # --- persist cards index and project_map atomically when not a dry run ---
        if not dry_run:
            # Write cards index
            try:
                cards_index_path = root / ".aidev" / "cards" / "index.json"
                written = _try_kb_write_cards_index(cards_index_path)
                if written:
                    # KB wrote it; try to read it back to get count
                    try:
                        text = cards_index_path.read_text(encoding="utf-8")
                        idx = json.loads(text)
                        index_count = len(idx) if isinstance(idx, list) else None
                    except Exception:
                        index_count = None
                else:
                    idx = _build_cards_index(filtered_paths)
                    _atomic_write_json(cards_index_path, idx)
                    index_count = len(idx)

                meta["cards_index_written"] = True
                meta["cards_index_written_ts"] = datetime.utcnow().isoformat() + "Z"
                try:
                    progress_cb(
                        "cards_index_written",
                        {
                            "path": str(cards_index_path),
                            "rec_id": rid,
                            "job_id": job_id,
                            "timestamp": meta.get("cards_index_written_ts"),
                            "index_count": index_count,
                        },
                    )
                except Exception:
                    logging.debug("Failed to emit cards_index_written event", exc_info=True)
            except Exception as e:
                meta["cards_index_written"] = False
                progress_error_cb(
                    "cards_index_write_failed",
                    {
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "rec_id": rid,
                        "job_id": job_id,
                    },
                )

            # Write project_map.json atomically if we have an in-memory pm or an existing file
            try:
                pm_out_path = root / ".aidev" / "project_map.json"
                if isinstance(pm, dict):
                    _atomic_write_json(pm_out_path, pm)
                elif pm_out_path.exists():
                    # Re-write existing project_map.json atomically for consistency
                    try:
                        text = pm_out_path.read_text(encoding="utf-8")
                        try:
                            obj = json.loads(text)
                        except Exception:
                            obj = text
                        _atomic_write_json(pm_out_path, obj)
                    except Exception:
                        # If we can't read existing, treat as not written but don't fail hard
                        pass

                meta["project_map_written"] = True
                meta["project_map_written_ts"] = datetime.utcnow().isoformat() + "Z"
                try:
                    progress_cb(
                        "project_map_written",
                        {
                            "path": str(pm_out_path),
                            "rec_id": rid,
                            "job_id": job_id,
                            "timestamp": meta.get("project_map_written_ts"),
                            "total_files": pm.get("total_files") if isinstance(pm, dict) else None,
                        },
                    )
                except Exception:
                    logging.debug("Failed to emit project_map_written event", exc_info=True)
            except Exception as e:
                meta["project_map_written"] = False
                progress_error_cb(
                    "project_map_write_failed",
                    {
                        "error": str(e),
                        "trace": traceback.format_exc(),
                        "rec_id": rid,
                        "job_id": job_id,
                    },
                )

    except Exception as e:
        # Overarching failure for refresh + persist; do not raise, just report
        progress_error_cb(
            "post_refresh_persist_failed",
            {
                "error": str(e),
                "trace": traceback.format_exc(),
                "rec_id": rid,
                "job_id": job_id,
            },
        )

    # Final structured done summary before return (additive only)
    try:
        try:
            project_root_str = str(project_root_canonical)
        except Exception:
            project_root_str = str(root)

        _safe_structured_info(
            "apply.done",
            meta={
                "rec_id": rid,
                "job_id": job_id,
                "changed_paths_count": len(filtered_paths),
                "created": created_count,
                "modified": modified_count,
                "skipped": skipped_count,
                "project_root": project_root_str,
                "dry_run": bool(dry_run),
                "refresh": {
                    "project_map_written": meta.get("project_map_written"),
                    "cards_index_written": meta.get("cards_index_written"),
                },
            },
            fallback_msg=(
                "[apply_and_refresh] done rec_id=%s job_id=%s changed=%d created=%d modified=%d skipped=%d" % (
                    rid, job_id, len(filtered_paths), created_count, modified_count, skipped_count
                )
            ),
        )
    except Exception:
        logging.debug("Failed to emit structured apply.done log", exc_info=True)

    return filtered_paths



# Compatibility wrapper: tolerant argument mapping and consistent return shape
def apply_and_refresh(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Wrapper around apply_edits_and_refresh that accepts common positional
    and keyword argument names used across the codebase/tests and always
    returns a dict-shaped result.

    Accepts flexible names/positions for: root (project_root, repo_root, project_path), rec_edits
    (applied_files, applied_files), applied_rec_ids, dry_run, st, kb,
    meta, stats_obj, progress_cb, progress_error_cb, job_id, writes_by_rec.

    Behavior:
      - If dry_run=True and the underlying call returns a dict, it is
        returned unchanged.
      - If dry_run=False and the underlying call returns a list of
        changed paths, the list is wrapped as {
            'changed_paths': [...],
            'refresh': {'ok': bool, 'meta': {...}}
        }
    """
    # Normalize positional args: commonly callers use (project_root, applied_files, dry_run)
    root = kwargs.get("root") or kwargs.get("project_root") or kwargs.get("repo_root") or kwargs.get("project_path")
    rec_edits = kwargs.get("rec_edits") or kwargs.get("applied_files")
    dry_run = kwargs.get("dry_run")

    if len(args) >= 1 and root is None:
        root = args[0]
    if len(args) >= 2 and rec_edits is None:
        rec_edits = args[1]
    if len(args) >= 3 and dry_run is None:
        dry_run = args[2]

    # Ensure sensible defaults
    if dry_run is None:
        dry_run = bool(kwargs.get("dry_run", False))

    # If caller did not explicitly supply rec_edits / applied_files,
    # this is a refresh-only invocation. Do NOT treat applied_rec_ids as edits.
    if rec_edits is None:
        rec_edits = []

    # Other common parameters
    st = kwargs.get("st")
    kb = kwargs.get("kb")
    meta = kwargs.get("meta") or {}
    stats_obj = kwargs.get("stats_obj")
    progress_cb = kwargs.get("progress_cb")
    progress_error_cb = kwargs.get("progress_error_cb")
    job_id = kwargs.get("job_id")
    writes_by_rec = kwargs.get("writes_by_rec")
    rec_id = kwargs.get("rec_id")
    selected = kwargs.get("selected", True)

    # Provide no-op callbacks when not supplied to avoid None checks in the callee
    def _noop(event: str, payload: Dict[str, Any]) -> None:
        return None

    if progress_cb is None:
        progress_cb = _noop
    if progress_error_cb is None:
        progress_error_cb = _noop

    # Ensure root is provided (the underlying function requires it)
    if root is None:
        raise TypeError("apply_and_refresh requires a 'root', 'project_root', 'repo_root', or 'project_path' argument")

    # Coerce string roots to Path for convenience
    if isinstance(root, str):
        try:
            root = Path(root)
        except Exception:
            # Let apply_edits_and_refresh surface the error
            pass

    # Prepare args for underlying function
    try:
        result = apply_edits_and_refresh(
            root=root,
            repo_root=kwargs.get("repo_root") or kwargs.get("project_root") or root,
            kb=kb,
            meta=meta,
            st=st,
            rec=(kwargs.get("rec") or {}),
            rec_edits=rec_edits,
            stats_obj=stats_obj,
            progress_cb=progress_cb,
            progress_error_cb=progress_error_cb,
            job_id=job_id,
            writes_by_rec=writes_by_rec,
            rec_id=rec_id,
            selected=selected,
            dry_run=dry_run,
        )
    except Exception:
        # Let exceptions propagate after emitting an error via progress_error_cb
        try:
            progress_error_cb(
                "apply_and_refresh_error",
                {
                    "error": traceback.format_exc(),
                    "rec_id": rec_id,
                    "job_id": job_id,
                },
            )
        except Exception:
            logging.debug("Failed to emit apply_and_refresh_error", exc_info=True)
        raise

    # If dry-run and the underlying function already returned a dict, return it unchanged
    if dry_run and isinstance(result, dict):
        return result

    # For non-dry-run, normalize the return into a dict with refresh metadata
    out: Dict[str, Any] = {}
    if isinstance(result, list):
        changed_paths = result
    elif isinstance(result, dict) and "changed_paths" in result:
        changed_paths = result.get("changed_paths")
    else:
        # Unexpected shape: coerce best-effort
        try:
            changed_paths = list(result)
        except Exception:
            changed_paths = []

    refresh_meta_keys = [
        "cards_index_written",
        "project_map_written",
        "cards_index_written_ts",
        "project_map_written_ts",
    ]
    refresh_meta: Dict[str, Any] = {k: meta.get(k) for k in refresh_meta_keys if k in meta}
    ok = True if "project_map_written" not in meta else bool(meta.get("project_map_written"))

    out["changed_paths"] = changed_paths
    out["refresh"] = {"ok": ok, "meta": refresh_meta}
    return out
