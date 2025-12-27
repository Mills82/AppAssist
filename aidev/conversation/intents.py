# aidev/conversation/intents.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import ChatGPT, system_preset

# ---------------- Intents & Slots ----------------


class Intent(str, Enum):
  CREATE_PROJECT       = "CREATE_PROJECT"
  SELECT_PROJECT       = "SELECT_PROJECT"
  UPDATE_DESCRIPTIONS  = "UPDATE_DESCRIPTIONS"
  RUN_CHECKS           = "RUN_CHECKS"

  Q_AND_A              = "Q_AND_A"            # "Explain this", "What does X do?"
  ANALYZE_PROJECT      = "ANALYZE_PROJECT"    # "Analyze my app structure", "Audit X"

  MAKE_RECOMMENDATIONS = "MAKE_RECOMMENDATIONS"
  APPLY_EDITS          = "APPLY_EDITS"


# Slot names we might produce for downstream planners/tools.
# (Planner will read these; all are optional.)
SLOT_KEYS = {
  "project_name",      # e.g., "my-app"
  "project_path",      # absolute or relative
  "base_dir",          # directory to create the project in
  "create_if_missing", # bool
  "focus",             # user goal / task description (planner will decide card policy)
  "instructions",      # text for update_descriptions
  "tech_stack",        # e.g., ["flutter", "react", "fastapi"]
  "framework",         # free-form; may duplicate tech_stack
  "model",             # model/deployment name if user specifies
  "focus_raw",         # exact chat text that triggered MAKE_RECOMMENDATIONS
  # planner currently reads these; allow them for LLM output
  "targets",           # list[str] (files/paths)
  "top_k",             # int (QA: KB selection size)
  "answers",           # dict or text (create flow) — schema includes this
  "answers_text",      # free-form answers text (create flow)
}


@dataclass
class IntentResult:
  """
  Unified intent result for conversation + planning.

  - intent: one of the Intent enum values
  - slots:  best-effort structured details for planner / orchestrator
  - confidence: classifier's confidence in [0, 1]
  - rationale:  short classifier explanation (free-form text)
  - matched_rules: optional debug info when rules fired (pattern string or label)
  """
  intent: Intent
  slots: Dict[str, Any]
  confidence: float = 0.66
  rationale: str = ""
  matched_rules: Optional[str] = None


# ---------------- Rule Hints ----------------

# NOTE: Order matters. Earlier rules win when multiple patterns match.
# Q_AND_A is intentionally placed earlier than MAKE_RECOMMENDATIONS / ANALYZE_PROJECT
# so direct question forms prefer Q&A when rules match.
_RULES: List[Tuple[Intent, re.Pattern[str]]] = [
  # CREATE_PROJECT
  (
    Intent.CREATE_PROJECT,
    re.compile(
      r"\b(create|new|scaffold|bootstrap|spin\s*up|generate)\b.*\b(app|project|site|website|starter|template|boilerplate)\b",
      re.I,
    ),
  ),
  # SELECT_PROJECT
  (
    Intent.SELECT_PROJECT,
    re.compile(
      r"\b(select|pick|choose|open|switch\s*to)\b.*\b(project)\b",
      re.I,
    ),
  ),
  # UPDATE_DESCRIPTIONS
  (
    Intent.UPDATE_DESCRIPTIONS,
    re.compile(
      r"\b(update|revise|rewrite|improve)\b.*\b(description|descriptions|overview|project\s+description)\b",
      re.I,
    ),
  ),
  # RUN_CHECKS
  (
    Intent.RUN_CHECKS,
    re.compile(
      r"\b(run|execute|perform)\b.*\b(checks?|lint|format|tests?)\b",
      re.I,
    ),
  ),
  # Q_AND_A — general “what / why / how / explain” questions
  (
    Intent.Q_AND_A,
    re.compile(
      r"(\b(what|why|how|where|when|who)\b)"
      r"|\bhow\s+to\b"
      r"|\bhow\s+do\s+i\b"
      r"|\bhow\s+can\s+i\b"
      r"|\b(can|could)\s+you\s+explain\b"
      r"|\bexplain\b"
      r"|\bdescribe\b"
      r"|\bwhat(?:'s| is)\b",
      re.I,
    ),
  ),
  # MAKE_RECOMMENDATIONS — explicitly about what to change/improve/next
  (
    Intent.MAKE_RECOMMENDATIONS,
    re.compile(
      r"\b("
      r"recommend(ations?)?"
      r"|roadmap"
      r"|next\s+steps?"
      r"|suggest(ion|ions)?"
      r"|what\s+should\s+I\s+(change|improve|do\s+next)"
      r"|how\s+can\s+I\s+(improve|refactor|clean\s+up)"
      r")\b",
      re.I,
    ),
  ),
  # ANALYZE_PROJECT — project/codebase analysis / audit
  (
    Intent.ANALYZE_PROJECT,
    re.compile(
      r"\b("
      r"analy[sz]e\b.*\b(project|code(base)?|repo|repository)"
      r"|audit\b.*\b(code|project)"
      r"|review\b.*\b(project|code)"
      r"|give\s+me\s+(an|a)\s+(overview|analysis)\s+of\b"
      r")",
      re.I,
    ),
  ),
  # APPLY_EDITS
  (
    Intent.APPLY_EDITS,
    re.compile(
      r"\b(apply|commit|merge)\b.*\b(change|changes|edits|patch(?:es)?)\b",
      re.I,
    ),
  ),
]


# ---------------- JSON Schema for LLM (loaded from file) ----------------

_INTENT_ENUM = [i.value for i in Intent]

_INTENT_SCHEMA_NAME = "intent_classification.schema.json"
_INTENT_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def _schema_dir() -> Path:
  # aidev/conversation/intents.py -> aidev/schemas/
  return Path(__file__).resolve().parent.parent / "schemas"


def _load_intent_schema() -> Dict[str, Any]:
  global _INTENT_SCHEMA_CACHE
  if _INTENT_SCHEMA_CACHE is not None:
    return _INTENT_SCHEMA_CACHE

  p = _schema_dir() / _INTENT_SCHEMA_NAME
  if not p.exists():
    raise FileNotFoundError(f"Missing schema file: {p}")

  _INTENT_SCHEMA_CACHE = json.loads(p.read_text(encoding="utf-8"))
  return _INTENT_SCHEMA_CACHE


# ---------------- Classifier ----------------


def _rules_first_guess(utterance: str) -> Optional[Tuple[Intent, re.Pattern[str]]]:
  """
  Returns (Intent, matching_pattern) if a rule/heuristic matches, otherwise None.

  Heuristic bias: prefer Q&A for question-like inputs (ends with '?', or contains 'how to').
  """
  utterance_stripped = (utterance or "").strip()

  if utterance_stripped.endswith("?") or "how to" in utterance_stripped.lower():
    return (Intent.Q_AND_A, re.compile("heuristic:question-mark-or-how-to"))

  for intent, pat in _RULES:
    if pat.search(utterance or ""):
      return (intent, pat)
  return None


def _filter_slots(slots: Any) -> Dict[str, Any]:
  """Keep only allowed keys and drop obvious empties."""
  if not isinstance(slots, dict):
    return {}
  out: Dict[str, Any] = {}
  for k, v in slots.items():
    if k not in SLOT_KEYS:
      continue
    if v is None:
      continue
    # drop empty strings / empty lists / empty dicts
    if isinstance(v, str) and not v.strip():
      continue
    if isinstance(v, list) and len(v) == 0:
      continue
    if isinstance(v, dict) and len(v) == 0:
      continue
    out[k] = v
  return out


def _normalize_str_list(x: Any) -> Optional[List[str]]:
  """Normalize a string or list into a cleaned list[str], or None if empty/unknown."""
  if x is None:
    return None
  if isinstance(x, list):
    items = [str(t).strip() for t in x if str(t).strip()]
    return items or None
  if isinstance(x, str):
    s = x.strip()
    if not s:
      return None
    # Split on commas, pluses, or newlines
    parts = re.split(r"[,\n\+]+", s)
    items = [p.strip() for p in parts if p.strip()]
    return items or None
  return None


def classify_intent_slots(
  utterance: str,
  *,
  history: Optional[List[str]] = None,
  projects: Optional[List[Dict[str, Any]]] = None,
  llm: Optional[ChatGPT] = None,
) -> IntentResult:
  """
  Returns an IntentResult. Uses quick rules first; then LLM JSON mode if provided.
  - `projects` can be a list of candidates (path, name, kind) to help SELECT_PROJECT.

  Guarantees:
  - For MAKE_RECOMMENDATIONS, preserve the user's words in focus_raw and ensure focus is non-empty when possible.
  - For UPDATE_DESCRIPTIONS, ensure instructions is non-empty when possible.
  - For Q_AND_A / ANALYZE_PROJECT, focus may be populated with the utterance for convenience.
  """
  rule_guess = _rules_first_guess(utterance)
  utterance_stripped = (utterance or "").strip()

  # Prefer LLM if provided (rules become a hint via rule_guess)
  if llm is not None:
    user_payload = {
      "utterance": utterance,
      "history": history or [],
      "projects": projects or [],
      "intent_options": _INTENT_ENUM,
      "slot_keys": sorted(list(SLOT_KEYS)),
      "rule_guess": (rule_guess[0].value if rule_guess else None),
    }
    system_text = system_preset("intent_classify")

    try:
      # Always send a string payload to the LLM client.
      user_text = json.dumps(user_payload, ensure_ascii=False)

      data, _res = llm.chat_json(
        [{"role": "user", "content": user_text}],
        schema=_load_intent_schema(),
        system=system_text,
      )

      if isinstance(data, dict):
        intent_str = data.get("intent")
        raw_slots = data.get("slots", {}) or {}
        slots = _filter_slots(raw_slots)

        # Normalize tech_stack to list[str]
        norm_tech = _normalize_str_list(slots.get("tech_stack"))
        if norm_tech is not None:
          slots["tech_stack"] = norm_tech

        # Normalize targets to list[str]
        norm_targets = _normalize_str_list(slots.get("targets"))
        if norm_targets is not None:
          slots["targets"] = norm_targets

        # Coerce intent_str to our enum
        intent_enum: Optional[Intent] = Intent(intent_str) if intent_str in _INTENT_ENUM else None

        # Strong guarantees about focus / instructions based on intent
        if intent_enum is Intent.MAKE_RECOMMENDATIONS and utterance_stripped:
          slots.setdefault("focus_raw", utterance_stripped)
          if not str(slots.get("focus", "")).strip():
            slots["focus"] = utterance_stripped

        elif intent_enum is Intent.UPDATE_DESCRIPTIONS and utterance_stripped:
          if not str(slots.get("instructions", "")).strip():
            slots["instructions"] = utterance_stripped

        elif intent_enum in (Intent.Q_AND_A, Intent.ANALYZE_PROJECT) and utterance_stripped:
          slots.setdefault("focus_raw", utterance_stripped)
          slots.setdefault("focus", utterance_stripped)

        conf = float(data.get("confidence", 0.66))
        rationale = str(data.get("rationale") or "") or "llm-classifier"
        matched_rules = data.get("matched_rules", None)
        if matched_rules is not None and not isinstance(matched_rules, str):
          matched_rules = str(matched_rules)

        if intent_enum is not None:
          return IntentResult(
            intent=intent_enum,
            slots=slots,
            confidence=max(0.0, min(1.0, conf)),
            rationale=rationale,
            matched_rules=matched_rules or (rule_guess[1].pattern if rule_guess else None),
          )
    except Exception:
      # Fall through to rules if LLM path fails
      pass

  # Rules-only path
  if rule_guess:
    intent, pattern = rule_guess
    slots: Dict[str, Any] = {}

    if intent is Intent.MAKE_RECOMMENDATIONS and utterance_stripped:
      slots["focus"] = utterance_stripped
      slots["focus_raw"] = utterance_stripped

    elif intent is Intent.UPDATE_DESCRIPTIONS and utterance_stripped:
      slots["instructions"] = utterance_stripped

    elif intent in (Intent.Q_AND_A, Intent.ANALYZE_PROJECT) and utterance_stripped:
      slots["focus"] = utterance_stripped
      slots["focus_raw"] = utterance_stripped

    confidence_map = {
      Intent.Q_AND_A: 0.75,
      Intent.ANALYZE_PROJECT: 0.65,
      Intent.CREATE_PROJECT: 0.85,
    }
    conf = confidence_map.get(intent, 0.55)
    rationale = f"rule-based: {intent.name} matched '{pattern.pattern}'"

    return IntentResult(
      intent=intent,
      slots=slots,
      confidence=conf,
      rationale=rationale,
      matched_rules=pattern.pattern,
    )

  # Default to MAKE_RECOMMENDATIONS with utterance as focus
  slots: Dict[str, Any] = {}
  if utterance_stripped:
    slots["focus"] = utterance_stripped
    slots["focus_raw"] = utterance_stripped

  return IntentResult(
    intent=Intent.MAKE_RECOMMENDATIONS,
    slots=slots,
    confidence=0.5,
    rationale="default",
  )


def detect_intent(
  utterance: str,
  *,
  history: Optional[List[str]] = None,
  projects: Optional[List[Dict[str, Any]]] = None,
  llm: Optional[ChatGPT] = None,
) -> IntentResult:
  """Primary entrypoint for chat + orchestration."""
  return classify_intent_slots(
    utterance,
    history=history,
    projects=projects,
    llm=llm,
  )


def detect_intent_label(utterance: str, *, llm: Optional[ChatGPT] = None) -> str:
  """Back-compat helper that returns just the intent label string."""
  res = detect_intent(utterance, llm=llm)
  return res.intent.value


def classify_intent(
  utterance: str,
  *,
  history: Optional[List[str]] = None,
  projects: Optional[List[Dict[str, Any]]] = None,
  llm: Optional[ChatGPT] = None,
) -> IntentResult:
  """Backwards-compatible wrapper expected by older callers/tests."""
  return classify_intent_slots(utterance, history=history, projects=projects, llm=llm)


# Backwards-compatible aliases some callers/tests expect to import by name.
classify = classify_intent
get_intent = classify_intent
infer_intent = classify_intent
intent_classifier = classify_intent


__all__ = [
  "Intent",
  "IntentResult",
  "SLOT_KEYS",
  "classify_intent_slots",
  "classify_intent",
  "detect_intent",
  "detect_intent_label",
]
