# aidev/mode_inference.py
"""Lightweight mode inference helper.

This module exposes infer_mode(...) which maps a freeform user message
to one of the high-level orchestration modes: "Q&A", "analyze", or "edit".

It is now a thin wrapper around the conversation intent classifier in
aidev.conversation.intents.detect_intent so that ALL intent logic lives
in a single place. This module exists mostly for CLI / legacy callers
that still want the simpler {mode, confidence} shape.

When running in auto mode (mode omitted or equal to "auto"), this helper:

  * calls detect_intent(...) to obtain an IntentResult
  * derives a canonical mode ("Q&A" | "analyze" | "edit") from the intent
  * writes a structured trace entry (event="intent_inference") using
    ProjectState trace helpers when available

On failure (e.g. detect_intent throws) it writes an error trace entry and
returns None instead of silently defaulting to a mode.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# Try to import the project's state/tracing helpers. If unavailable, tracing is
# simply skipped but inference still returns a result.
try:
  from aidev.state import ProjectState
except Exception:  # pragma: no cover - defensive
  ProjectState = None  # type: ignore

# Unified conversation intents: keep imports lazy to avoid circular import issues.
# Importing aidev.conversation.intents at module import time can cause circular
# import problems in test environments; therefore we lazily load the module
# when needed.
Intent = None  # type: ignore
IntentResult = None  # type: ignore
detect_intent = None  # type: ignore


def _ensure_intent_imports() -> None:
  """Attempt to import Intent, IntentResult, detect_intent lazily.

  This will silently leave the names as None if the module or attributes
  aren't available; callers must handle the None case defensively.
  """
  global Intent, IntentResult, detect_intent
  # If already loaded, do nothing.
  if detect_intent is not None and Intent is not None:
    return
  try:
    mod = __import__("aidev.conversation.intents", fromlist=["Intent", "IntentResult", "detect_intent"])
    Intent = getattr(mod, "Intent", None)
    IntentResult = getattr(mod, "IntentResult", None)
    detect_intent = getattr(mod, "detect_intent", None)
  except Exception:
    # Keep defensive behavior: leave names as None when import fails.
    Intent = None
    IntentResult = None
    detect_intent = None


_ALLOWED_MODES = ("Q&A", "analyze", "edit")

# Thresholds and heuristics for conservative Q&A bias
_NON_QA_MIN_CONFIDENCE = 0.6
_ANALYZE_MIN_CONFIDENCE = 0.6
_QA_BIAS_FLOOR = 0.65
_QUESTION_RE = re.compile(r"^(what|why|how|where|when|who)\b|\bhow to\b|\bhow do i\b|\bexplain\b", re.I)


def _normalize_mode(raw: str) -> Optional[str]:
  """Normalize freeform mode text to one of the allowed canonical modes.

  Returns None when the raw text can't be mapped.
  """
  if not raw:
    return None
  s = raw.strip().lower()
  # common variants
  if s in ("q&a", "qa", "q_and_a", "question", "question-answer", "question_answer"):
    return "Q&A"
  if "qa" in s and "an" not in s:
    return "Q&A"
  if "analy" in s or "inspect" in s or "summar" in s:
    return "analyze"
  if "edit" in s or "fix" in s or "refactor" in s or "change" in s:
    return "edit"
  # exact matches of canonical forms
  for m in _ALLOWED_MODES:
    if s == m.lower():
      return m
  return None


def _safe_confidence(val: Any) -> float:
  """Coerce various confidence representations to a float in [0.0, 1.0]."""
  try:
    f = float(val)
  except Exception:
    return 0.0
  # if given as percentage >1, assume percent
  if f > 1.0:
    try:
      # clamp percentage to [0,100]
      if f <= 100.0:
        return max(0.0, min(1.0, f / 100.0))
    except Exception:
      pass
  return max(0.0, min(1.0, f))


def _write_trace_entry(entry: Dict[str, Any]) -> None:
  """Best-effort write the trace entry using ProjectState trace helpers.

  This function will not raise; failures are logged at debug level so that
  inference still returns a result even if tracing is unavailable.
  """
  if ProjectState is None:
    LOGGER.debug("ProjectState unavailable; skipping trace write")
    return

  try:
    # Preferred: a shared trace logger instance available at ProjectState.trace
    trace = getattr(ProjectState, "trace", None)
    if trace is not None and hasattr(trace, "write"):
      trace.write(entry)
      return

    # Fallback: a TraceLogger class on ProjectState we can instantiate
    TraceLogger = getattr(ProjectState, "TraceLogger", None)
    if TraceLogger is not None:
      tl = TraceLogger()
      if hasattr(tl, "write"):
        tl.write(entry)
        return

    LOGGER.debug("No trace writer found on ProjectState; trace entry not written")
  except Exception:
    LOGGER.debug("Failed to write trace entry", exc_info=True)


def _mode_from_intent(intent: Any) -> str:
  """
  Map a high-level Intent enum to a canonical run mode.

  - MAKE_RECOMMENDATIONS / APPLY_EDITS / CREATE/SELECT_PROJECT → "edit"
  - RUN_CHECKS / UPDATE_DESCRIPTIONS                            → "analyze"
  - Anything else                                              → "Q&A"
  """
  # Ensure intent types are loaded if possible (lazy import)
  _ensure_intent_imports()

  # If the Intent enum isn't available for some reason, default to Q&A.
  if Intent is None:
    # Accept some string-like intents defensively
    try:
      name = str(intent).lower()
      # Prefer explicit analyze-like substrings early
      if any(k in name for k in ("analy", "inspect", "run_analyze", "analysis")):
        return "analyze"
      if any(k in name for k in ("make_recommendations", "apply_edits", "create_project", "select_project", "edit", "fix", "refactor")):
        return "edit"
      if any(k in name for k in ("run_checks", "update_descriptions", "analyze", "inspect", "summar")):
        return "analyze"
    except Exception:
      return "Q&A"

  try:
    # Safe construction of analyze-like intent set from enum attributes
    analyze_intents = []
    for attr in ("RUN_CHECKS", "UPDATE_DESCRIPTIONS", "ANALYZE", "INSPECT", "RUN_ANALYZE"):
      if hasattr(Intent, attr):
        try:
          analyze_intents.append(getattr(Intent, attr))
        except Exception:
          # be defensive, skip problematic attributes
          pass

    # Edit-like intents
    edit_intents = []
    for attr in ("MAKE_RECOMMENDATIONS", "APPLY_EDITS", "CREATE_PROJECT", "SELECT_PROJECT"):
      if hasattr(Intent, attr):
        try:
          edit_intents.append(getattr(Intent, attr))
        except Exception:
          pass

    if intent in tuple(edit_intents):
      return "edit"

    if intent in tuple(analyze_intents):
      return "analyze"
  except Exception:
    # Very defensive: if membership checks blow up, fall back to string-based checks
    try:
      name = str(intent).lower()
      if any(k in name for k in ("analy", "inspect", "run_analyze", "analysis")):
        return "analyze"
      if any(k in name for k in ("make_recommendations", "apply_edits", "create_project", "select_project", "edit", "fix", "refactor")):
        return "edit"
      if any(k in name for k in ("run_checks", "update_descriptions", "analyze", "inspect", "summar")):
        return "analyze"
    except Exception:
      pass
  return "Q&A"


def infer_mode(
  message_text: str,
  *,
  mode: Optional[str] = None,
  model: Optional[str] = None,
  message_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
  """Infer a mode for the given message text.

  When `mode` is provided and is not the special value "auto" (case-insensitive)
  the function returns that override (normalized) without calling the intent
  classifier.

  When `mode` is omitted or equals "auto" the function will:

    * call conversation.intents.detect_intent(...) to obtain an IntentResult
    * map that intent to a canonical mode ("Q&A", "analyze", "edit")
    * write an 'intent_inference' trace entry (best-effort)
    * return { "mode": <canonical>, "confidence": <float in [0,1]> }

  On classifier failure the function writes an error trace entry and returns
  None instead of silently defaulting to a mode.
  """
  # If a concrete override is provided and not 'auto', honor it and do not call LLM/intent
  if mode is not None and str(mode).strip().lower() != "auto":
    normalized = _normalize_mode(str(mode))
    if normalized:
      # keep backward-compatible shape but also include intent/rationale placeholders
      return {"mode": normalized, "confidence": 1.0, "intent": None, "rationale": "explicit_override"}
    # Invalid explicit override: return None to signal the caller
    return None

  # Ensure intent helpers are imported lazily to avoid circular import problems
  _ensure_intent_imports()

  start_ts = time.time()
  err_code: Optional[str] = None
  err_msg: Optional[str] = None
  intent_res: Optional[Any] = None

  # The function may be called with an IntentResult-like object (from classifier_fn).
  # Accept a dict or object that contains an 'intent' and/or 'confidence' attribute
  # and skip calling detect_intent in that case.
  if not isinstance(message_text, str):
    if isinstance(message_text, dict) and ("intent" in message_text or "confidence" in message_text):
      intent_res = message_text
    elif hasattr(message_text, "intent") or hasattr(message_text, "confidence"):
      intent_res = message_text

  # Auto-mode: rely on the central intent detector if we don't already have an IntentResult
  if intent_res is None:
    if detect_intent is None:
      err_code = "detect_intent_unavailable"
      err_msg = "conversation.intents.detect_intent is not available"
    else:
      try:
        intent_res = detect_intent(message_text)
      except Exception as e:  # pragma: no cover - defensive
        LOGGER.debug("detect_intent raised an exception", exc_info=True)
        err_code = "detect_intent_exception"
        err_msg = str(e)

  latency_ms = int((time.time() - start_ts) * 1000)

  if err_code is not None or intent_res is None:
    trace_entry = {
      "ts": datetime.utcnow().isoformat() + "Z",
      "event": "intent_inference",
      "user_message_id": message_id,
      "error_code": err_code or "unknown",
      "error_message": err_msg or "intent detection failed",
      "model": model,
      "latency_ms": latency_ms,
      "rationale": "detection_failed",
    }
    _write_trace_entry(trace_entry)
    return None

  # Helper to access either dict-like or attribute-like intent_res
  def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
      return obj.get(key, default)
    return getattr(obj, key, default)

  # Defensive extraction of intent name and confidence
  try:
    intent_obj = _get(intent_res, "intent", None)
    intent_name_raw = None
    if isinstance(intent_obj, dict):
      intent_name_raw = intent_obj.get("name") or intent_obj.get("value")
    else:
      intent_name_raw = getattr(intent_obj, "name", None) or getattr(intent_obj, "value", None)

    intent_name = str(intent_name_raw if intent_name_raw is not None else intent_obj).lower()
  except Exception:
    intent_name = str(_get(intent_res, "intent", "")).lower()

  confidence = _safe_confidence(_get(intent_res, "confidence", None))
  canonical_mode = _mode_from_intent(_get(intent_res, "intent", None))

  # Text heuristics: detect obvious question-like inputs
  text = (message_text or "").strip()
  text_question = bool(_QUESTION_RE.search(text) or text.endswith("?"))

  # Intent-based question detection (defensive string checks)
  intent_question = any(k in (intent_name or "") for k in ("question", "qanda", "q_and_a", "qa", "q&a"))

  rationale = f"intent_to_{canonical_mode.lower()}"

  # Unit-test-friendly snippet (keep <=10 lines):
  # if classifier indicates analyze, prefer analyze deterministically
  # This must run before question/text heuristics so analyze is not silently
  # overridden by a surface-level question heuristic.
  if "analy" in (intent_name or "") or "inspect" in (intent_name or ""):
    canonical_mode = "analyze"
    rationale = "intent_analyze" if confidence >= _ANALYZE_MIN_CONFIDENCE else "intent_analyze_low_confidence"

  # Bias rules:
  # 1) If classifier intent clearly indicates question intent, prefer Q&A and boost confidence slightly.
  # NOTE: Do not override analyze intent with question heuristics.
  if canonical_mode != "analyze" and intent_question and canonical_mode != "Q&A":
    canonical_mode = "Q&A"
    rationale = "intent_question"
    confidence = max(confidence, _QA_BIAS_FLOOR)
  # 2) If textual heuristics look like a question, prefer Q&A.
  elif canonical_mode != "analyze" and text_question and canonical_mode != "Q&A":
    canonical_mode = "Q&A"
    rationale = "text_question"
    confidence = max(confidence, _QA_BIAS_FLOOR)
  # 3) If the intent mapped to a non-QA mode but confidence is low, fall back to Q&A conservatively.
  # NOTE: Do not silently fall back when analyze is indicated; return analyze with low confidence.
  elif canonical_mode != "analyze" and canonical_mode != "Q&A" and confidence < _NON_QA_MIN_CONFIDENCE:
    rationale = "low_confidence_fallback"
    canonical_mode = "Q&A"
    confidence = max(confidence, _QA_BIAS_FLOOR)
  else:
    # Keep canonical mapping rationale (intent_to_<mode>)
    rationale = rationale

  # Build trace and return structure. Keep back-compat keys in the trace (mode + inferred_mode).
  try:
    intent_obj = _get(intent_res, "intent", None)
    if isinstance(intent_obj, dict):
      intent_display = intent_obj.get("value") or intent_obj.get("name") or str(intent_obj)
    else:
      intent_display = getattr(intent_obj, "value", str(_get(intent_res, "intent", intent_name)))
  except Exception:
    intent_display = str(_get(intent_res, "intent", intent_name))

  try:
    slots_raw = _get(intent_res, "slots", {}) or {}
    slots = dict(slots_raw)
  except Exception:
    slots = {}

  trace_entry = {
    "ts": datetime.utcnow().isoformat() + "Z",
    "event": "intent_inference",
    "user_message_id": message_id,
    "intent": intent_display,
    "intent_name": intent_name,
    "mode": canonical_mode,
    "inferred_mode": canonical_mode,  # keep both keys for back-compat
    "confidence": confidence,
    "rationale": rationale,
    "model": model,
    "latency_ms": latency_ms,
    "slots": slots,
  }
  _write_trace_entry(trace_entry)

  return {"mode": canonical_mode, "confidence": confidence, "intent": intent_name, "rationale": rationale}
