# aidev/stages/quality_gates.py
from __future__ import annotations

import base64
import hashlib
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..runtime import detect_runtimes
from ..state import ProjectState
from .. import events as _events
from .consistency_checks import run_consistency_checks


ErrorFn = Callable[[str, Dict[str, Any]], None]


def _normalize_cfg(cfg: Any) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _encode_bytes_for_trace(b: Optional[bytes]) -> Optional[Dict[str, Any]]:
    if b is None:
        return None
    try:
        text = b.decode("utf-8")
        return {"text": text}
    except Exception:
        return {"b64": base64.b64encode(b).decode("ascii")}


def run_quality_gates(
    *,
    root: Path,
    st: ProjectState,
    cfg: Any,
    session_id: Optional[str],
    job_id: Optional[str],
    progress_error_cb: ErrorFn,
) -> None:
    """
    Run post-apply quality gates: format / lint / tests for detected runtimes,
    plus lightweight cross-file consistency checks.

    This is a refactor of Orchestrator._run_quality_gates into a standalone
    stage helper. It writes into the shared trace and emits the same
    _events.checks_* events so the UI behavior is unchanged, with additional
    "consistency" step results included in the payload.
    """
    _events.checks_started(total=None, session_id=session_id, job_id=job_id)

    try:
        runtimes = detect_runtimes(root)
    except Exception:
        logging.exception("detect_runtimes failed")
        try:
            st.trace.write("VERIFY", "error", {"error": "detect_runtimes failed"})
        except Exception:
            pass
        progress_error_cb(
            "detect_runtimes",
            error="detect_runtimes failed",
            job_id=job_id,
        )
        _events.checks_result(
            ok=False,
            results=[{"error": "detect_runtimes failed"}],
            session_id=session_id,
            job_id=job_id,
        )
        return

    cfg_dict = _normalize_cfg(cfg)
    qg = cfg_dict.get("quality_gates", {}) if isinstance(cfg_dict.get("quality_gates"), dict) else {}

    def _enabled(rt_name: str, step: str) -> bool:
        try:
            rt_key = rt_name.lower().replace("tools", "").strip("_")
            flags = qg.get(rt_key, {})
            if isinstance(flags, dict) and step in {"format", "lint", "test"}:
                return bool(flags.get(step, True))
        except Exception:
            pass
        return True

    results: List[Dict[str, Any]] = []

    def _maybe_call(rt: Any, step: str, name_variants) -> Optional[Dict[str, Any]]:
        for nm in name_variants:
            fn = getattr(rt, nm, None)
            if callable(fn):
                try:
                    with st.trace.timer(
                        "tool_step",
                        "runtime",
                        phase="quality_gates",
                        tool=getattr(rt, "name", type(rt).__name__),
                        step=step,
                    ):
                        t0 = time.time()
                        out = fn()
                        dt = round(time.time() - t0, 3)
                    return {
                        "tool": getattr(rt, "name", type(rt).__name__),
                        "step": step,
                        "elapsed_sec": dt,
                        "result": out,
                    }
                except Exception as e:
                    logging.debug(
                        "Runtime %s.%s failed: %s",
                        type(rt).__name__,
                        nm,
                        e,
                    )
                    progress_error_cb(
                        "quality_gate",
                        error=str(e),
                        step=step,
                        tool=getattr(rt, "name", type(rt).__name__),
                        trace=traceback.format_exc(),
                        job_id=job_id,
                    )
                    return {
                        "tool": getattr(rt, "name", type(rt).__name__),
                        "step": step,
                        "error": str(e),
                    }
        return None

    # 1) Standard runtime format / lint / test steps
    for rt in (runtimes or []):
        rt_name = getattr(rt, "name", type(rt).__name__)
        for step, aliases in (
            ("format", ("format", "run_format")),
            ("lint", ("lint", "run_lint")),
            ("test", ("run_tests", "test")),
        ):
            if not _enabled(rt_name, step):
                continue
            rec = _maybe_call(rt, step, aliases)
            if rec:
                results.append(rec)

    # 2) Cross-file consistency checks (HTML/JS/CSS etc.)
    # Config shape (optional) under cfg["consistency_checks"], e.g.:
    # {
    #   "enabled": true,
    #   "html_globs": ["aidev/ui/**/*.html"],
    #   "css_globs": ["aidev/ui/**/*.css"],
    #   "js_globs": ["aidev/ui/**/*.js"],
    #   "ignore_dirs": ["node_modules", ".git"],
    #   "max_issues": 100,
    #   # optional overlay of updated file contents to use for checks (useful
    #   # when doing a pre-apply repair flow where in-memory/updated content
    #   # should be checked instead of on-disk files):
    #   "file_overlays": {"path/relative/to/root.js": "updated content"}
    # }
    cc_cfg_raw = cfg_dict.get("consistency_checks")
    cc_cfg = cc_cfg_raw if isinstance(cc_cfg_raw, dict) else {}
    if cc_cfg.get("enabled", True):
        # Support an optional "file_overlays" map to temporarily write updated
        # content to disk so run_consistency_checks operates on the updated
        # file bytes rather than the on-disk fallback. This helps pre-apply
        # repair flows use the latest content when checking.
        overlays = cc_cfg.get("file_overlays")
        backups: Dict[str, Optional[bytes]] = {}
        created_paths: List[str] = []

        # For trace events: capture baseline bytes (if present) and the overlay
        # contents written so tests / UI can show diffs.
        baseline_map: Dict[str, Any] = {}
        overlay_map: Dict[str, Any] = {}

        try:
            # Apply overlays if provided
            if isinstance(overlays, dict) and overlays:
                for rel_path, content in overlays.items():
                    try:
                        target = (root / rel_path).resolve()
                        # Only allow targets under the root for safety
                        try:
                            target.relative_to(root.resolve())
                        except Exception:
                            raise ValueError(f"Overlay path {rel_path} is outside root")

                        if target.exists():
                            try:
                                orig_bytes = target.read_bytes()
                                backups[str(target)] = orig_bytes
                            except Exception:
                                orig_bytes = None
                                backups[str(target)] = None
                        else:
                            orig_bytes = None
                            backups[str(target)] = None
                            created_paths.append(str(target))
                            target.parent.mkdir(parents=True, exist_ok=True)

                        data: bytes
                        if isinstance(content, str):
                            data = content.encode("utf-8")
                        elif isinstance(content, (bytes, bytearray)):
                            data = bytes(content)
                        else:
                            data = str(content).encode("utf-8")

                        # Record baseline and overlay for tracing
                        baseline_map[rel_path] = _encode_bytes_for_trace(orig_bytes)
                        overlay_map[rel_path] = _encode_bytes_for_trace(data)

                        target.write_bytes(data)
                    except Exception as e:
                        logging.debug("Failed to write overlay for %s: %s", rel_path, e, exc_info=True)
                        progress_error_cb(
                            "consistency_overlay_write",
                            error=str(e),
                            file=rel_path,
                            job_id=job_id,
                        )

                # Emit trace events for baseline and overlay contents so
                # consumers can show diffs or diagnostics without needing to
                # read the FS directly.
                try:
                    st.trace.write(
                        "trace",
                        "baseline",
                        {
                            "files": baseline_map,
                            "session_id": session_id,
                            "job_id": job_id,
                        },
                    )
                except Exception:
                    logging.debug("Failed to write trace.baseline", exc_info=True)

                try:
                    st.trace.write(
                        "trace",
                        "overlay",
                        {
                            "files": overlay_map,
                            "session_id": session_id,
                            "job_id": job_id,
                        },
                    )
                except Exception:
                    logging.debug("Failed to write trace.overlay", exc_info=True)

                # Also emit compact, best-effort events via aidev.events helpers
                # so event-bus consumers receive the same run-level summaries
                # without duplicating raw file bytes. These calls are
                # best-effort and will not interrupt the quality gate flow.
                try:
                    baseline_summary: Dict[str, Any] = {}
                    overlay_summary: Dict[str, Any] = {}
                    for rel_path, content in overlays.items():
                        # overlay bytes as written
                        if isinstance(content, str):
                            overlay_bytes = content.encode("utf-8")
                        elif isinstance(content, (bytes, bytearray)):
                            overlay_bytes = bytes(content)
                        else:
                            overlay_bytes = str(content).encode("utf-8")

                        abs_target = str((root / rel_path).resolve())
                        orig_bytes = backups.get(abs_target)

                        def summarize(b: Optional[bytes]) -> Optional[Dict[str, Any]]:
                            if b is None:
                                return None
                            h = hashlib.sha256(b).hexdigest()
                            return {"sha256": h, "size": len(b)}

                        baseline_summary[rel_path] = summarize(orig_bytes)
                        overlay_summary[rel_path] = summarize(overlay_bytes)

                    # Call event helpers if available; do not assume they exist
                    try:
                        if hasattr(_events, "trace_baseline"):
                            _events.trace_baseline(
                                {"files": baseline_summary},
                                session_id=session_id,
                                job_id=job_id,
                            )
                    except Exception:
                        logging.debug("_events.trace_baseline failed", exc_info=True)

                    try:
                        if hasattr(_events, "trace_overlay"):
                            _events.trace_overlay(
                                {"files": overlay_summary},
                                session_id=session_id,
                                job_id=job_id,
                            )
                    except Exception:
                        logging.debug("_events.trace_overlay failed", exc_info=True)

                except Exception:
                    logging.debug("Failed to emit compact baseline/overlay event summaries", exc_info=True)

            # Emit the inputs we will use for consistency checks (cfg + overlays)
            try:
                st.trace.write(
                    "trace",
                    "check_inputs",
                    {
                        "cfg": cc_cfg,
                        "overlay_files": list(overlays.keys()) if isinstance(overlays, dict) else [],
                        "session_id": session_id,
                        "job_id": job_id,
                    },
                )
            except Exception:
                logging.debug("Failed to write trace.check_inputs", exc_info=True)

            # Also emit a compact check_inputs event via helpers so bus consumers
            # can correlate with baseline/overlay summaries. Keep payload small.
            try:
                try:
                    overlay_paths = list(overlays.keys()) if isinstance(overlays, dict) else []
                    if hasattr(_events, "trace_check_inputs"):
                        _events.trace_check_inputs(
                            {
                                "cfg_keys": sorted(list(cc_cfg.keys())),
                                "overlay_files": overlay_paths,
                                "session_id": session_id,
                                "job_id": job_id,
                            }
                        )
                except Exception:
                    logging.debug("_events.trace_check_inputs failed", exc_info=True)
            except Exception:
                # Extra guarded catch in case of unexpected attribute errors
                logging.debug("Failed to emit compact check_inputs event", exc_info=True)

            with st.trace.timer(
                "tool_step",
                "consistency_checks",
                phase="quality_gates",
                tool="consistency_checks",
                step="consistency",
            ):
                t0 = time.time()
                cc_result = run_consistency_checks(root=root, cfg=cc_cfg)
                dt = round(time.time() - t0, 3)

            rec: Dict[str, Any] = {
                "tool": "consistency_checks",
                "step": "consistency",
                "elapsed_sec": dt,
                "result": cc_result,
            }

            # If issues are present, treat this as a failing gate, but do not
            # lose the structured issue details.
            issues = cc_result.get("issues") or []
            if issues:
                rec["error"] = f"{len(issues)} consistency issue(s) detected"

            results.append(rec)
        except Exception as e:
            logging.debug(
                "Consistency checks failed: %s",
                e,
                exc_info=True,
            )
            progress_error_cb(
                "quality_gate",
                error=str(e),
                step="consistency",
                tool="consistency_checks",
                trace=traceback.format_exc(),
                job_id=job_id,
            )
            results.append(
                {
                    "tool": "consistency_checks",
                    "step": "consistency",
                    "error": str(e),
                }
            )
        finally:
            # Restore any backed-up files or remove created overlay files
            if isinstance(overlays, dict) and overlays:
                for rel_path in overlays.keys():
                    try:
                        target = (root / rel_path).resolve()
                        # Only restore paths we attempted
                        key = str(target)
                        if key in backups:
                            original = backups.get(key)
                            if original is None:
                                # File didn't exist before overlay -> remove it
                                try:
                                    if target.exists():
                                        target.unlink()
                                except Exception as e:
                                    logging.debug("Failed to remove overlay file %s: %s", target, e, exc_info=True)
                                    progress_error_cb(
                                        "consistency_overlay_restore",
                                        error=str(e),
                                        file=rel_path,
                                        job_id=job_id,
                                    )
                            else:
                                try:
                                    target.write_bytes(original)
                                except Exception as e:
                                    logging.debug("Failed to restore original file %s: %s", target, e, exc_info=True)
                                    progress_error_cb(
                                        "consistency_overlay_restore",
                                        error=str(e),
                                        file=rel_path,
                                        job_id=job_id,
                                    )
                    except Exception:
                        # Catch-all to ensure restore attempts don't raise
                        logging.debug("Unexpected error while restoring overlay %s", rel_path, exc_info=True)

    # 3) Trace + checks_result event
    try:
        st.trace.write("VERIFY", "runtimes", {"results": results})
    except Exception:
        logging.debug("Failed to write VERIFY.runtimes trace", exc_info=True)

    ok = all("error" not in r for r in results) if results else True
    _events.checks_result(
        ok=ok,
        results=results,
        session_id=session_id,
        job_id=job_id,
    )
