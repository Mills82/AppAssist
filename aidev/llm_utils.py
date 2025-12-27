# aidev/llm_utils.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def extract_json_block(text: str) -> str:
    """
    Best-effort extraction of the JSON-looking region from an LLM response.

    - Prefers ```json``` fenced blocks.
    - Otherwise, finds the first '[' or '{' and returns from there.
    """
    if not text:
        return ""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        return fenced.group(1).strip()
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    return text[start:].strip() if start is not None else text.strip()


@dataclass(frozen=True)
class ParseError:
    """Deterministic, user-displayable parse failure for LLM JSON outputs."""

    message: str
    offset: Optional[int]
    snippet: str
    raw_block: str
    suggestion: str


def _trim_to_balanced_json(block: str) -> str:
    """Trim a candidate JSON block to the first balanced top-level object/array.

    This helps when the model appends trailing commentary after valid JSON.
    If we cannot confidently find a balanced end, returns the original block.
    """
    if not block:
        return block

    # Find first top-level opener.
    start = None
    opener = None
    for i, ch in enumerate(block):
        if ch in "[{":
            start = i
            opener = ch
            break
    if start is None or opener is None:
        return block

    closer = "]" if opener == "[" else "}"

    depth = 0
    in_str = False
    escape = False

    for j in range(start, len(block)):
        ch = block[j]

        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return block[start : j + 1].strip()

        # Handle nested opposite bracket types too.
        if ch in "[{" and ch != opener:
            depth += 1
        elif ch in "]}" and ch != closer:
            depth -= 1

        if depth < 0:
            break

    return block.strip()


def _make_snippet(raw_block: str, offset: Optional[int]) -> str:
    if not raw_block:
        return ""
    if offset is None:
        # Deterministic fallback: show the beginning.
        head = raw_block[:80]
        return head + ("…" if len(raw_block) > 80 else "")

    off = max(0, min(int(offset), max(0, len(raw_block) - 1)))
    radius = 40
    start = max(0, off - radius)
    end = min(len(raw_block), off + radius)
    snippet = raw_block[start:end]
    caret_pos = off - start
    caret_line = " " * caret_pos + "^"
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(raw_block) else ""
    return f"{prefix}{snippet}{suffix}\n{caret_line}"


def parse_analyze_plan_text(text: str) -> Union[dict, list, ParseError]:
    """Parse analyze-plan-like LLM output into JSON or return a ParseError.

    Never raises on malformed input.
    """
    block = extract_json_block(text)
    trimmed = _trim_to_balanced_json(block)

    try:
        data = json.loads(trimmed)
        if isinstance(data, (dict, list)):
            return data
        return ParseError(
            message="Expected a top-level JSON object or array.",
            offset=None,
            snippet=_make_snippet(trimmed, None),
            raw_block=trimmed,
            suggestion="Ensure output is strict JSON with a single top-level object or array.",
        )
    except json.JSONDecodeError as e:
        # Deterministic, user-visible diagnostics; no stack traces.
        offset = getattr(e, "pos", None)
        err = ParseError(
            message="Invalid JSON in model output.",
            offset=offset,
            snippet=_make_snippet(trimmed, offset),
            raw_block=trimmed,
            suggestion=(
                "Ensure output is strict JSON only (no trailing text), and use a single top-level object or array."
            ),
        )
        logging.debug(
            "ParseError while parsing analyze plan JSON (offset=%s, msg=%s)",
            err.offset,
            str(e),
        )
        return err
    except Exception:
        # Extremely defensive: still deterministic.
        err = ParseError(
            message="Failed to parse JSON from model output.",
            offset=None,
            snippet=_make_snippet(trimmed, None),
            raw_block=trimmed,
            suggestion="Ensure output is strict JSON only (no extra text) and is well-formed.",
        )
        logging.debug("ParseError while parsing analyze plan JSON (non-JSONDecodeError)")
        return err


def parse_json_array(text: str) -> Optional[List[Any]]:
    """
    Parse an LLM response as a JSON array, returning None on failure.
    """
    parsed = parse_analyze_plan_text(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, ParseError):
        logging.debug(
            "Failed to parse JSON array from LLM output: %s (offset=%s)",
            parsed.message,
            parsed.offset,
        )
    return None


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse an LLM response as a JSON object, returning None on failure.
    """
    parsed = parse_analyze_plan_text(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, ParseError):
        logging.debug(
            "Failed to parse JSON object from LLM output: %s (offset=%s)",
            parsed.message,
            parsed.offset,
        )
    return None


def normalize_recommendations(payload: Any) -> List[Dict[str, Any]]:
    """
    Normalize a variety of LLM response shapes into a list of recommendations.

    Target shape (aligned with recommendations.schema.json v2):

        [
          {
            "schema_version": 2,
            "id": "rec-1",
            "title": str,
            "rationale": str,
            "reason": str,
            "summary": str,
            "risk": "low" | "medium" | "high",
            "acceptance_criteria": [str, ...],
            "budget_estimate": {...},
            "files": [...],
            "actions": [...]
          },
          ...
        ]

    Accepts:
    - A direct list of dicts.
    - An object with `rationales`, `recommendations`, or `items` keys.
    - Items that use `why` instead of `reason`, or describe acceptance inside `actions`.
    - Older, looser shapes and fills in missing fields with reasonable defaults.
    """
    out: List[Dict[str, Any]] = []

    def _norm_risk(value: Any) -> str:
        r = (str(value or "")).strip().lower()
        return r if r in ("low", "medium", "high") else "medium"

    def _norm_files(files_val: Any) -> List[Dict[str, Any]]:
        files_out: List[Dict[str, Any]] = []
        if not isinstance(files_val, list):
            return files_out
        for f in files_val:
            if not isinstance(f, dict):
                continue
            path = (f.get("path") or f.get("file") or "").strip()
            if not path:
                continue
            try:
                added = int(f.get("added") or 0)
            except Exception:
                added = 0
            try:
                removed = int(f.get("removed") or 0)
            except Exception:
                removed = 0
            why = (f.get("why") or f.get("reason") or "").strip()
            files_out.append(
                {
                    "path": path,
                    "added": max(0, added),
                    "removed": max(0, removed),
                    "why": why,
                }
            )
        return files_out

    def _norm_actions(actions_val: Any) -> List[Dict[str, Any]]:
        actions_out: List[Dict[str, Any]] = []
        if not isinstance(actions_val, list):
            return actions_out
        for a in actions_val:
            if not isinstance(a, dict):
                continue
            typ = (a.get("type") or "").strip()
            if not typ:
                continue
            summary = (a.get("summary") or "").strip()
            if not summary:
                continue
            path = (a.get("path") or "").strip()
            why = (a.get("why") or "").strip()
            acceptance = (a.get("acceptance") or "").strip()
            refs_raw = a.get("references") or []
            refs: List[str] = []
            if isinstance(refs_raw, list):
                for r in refs_raw:
                    s = (str(r) or "").strip()
                    if s:
                        refs.append(s)
            risk_val = _norm_risk(a.get("risk"))
            action_obj: Dict[str, Any] = {
                "type": typ,
                "summary": summary,
            }
            if path:
                action_obj["path"] = path
            if why:
                action_obj["why"] = why
            if refs:
                action_obj["references"] = refs
            if acceptance:
                action_obj["acceptance"] = acceptance
            if risk_val:
                action_obj["risk"] = risk_val
            actions_out.append(action_obj)
        return actions_out

    def _ensure_acceptance(item: Dict[str, Any]) -> List[str]:
        # Prefer explicit acceptance_criteria.
        ac = item.get("acceptance_criteria")
        if ac is None and isinstance(item.get("actions"), list):
            # Sometimes acceptance criteria live inside `actions`.
            derived: List[str] = []
            for a in item["actions"]:
                if not isinstance(a, dict):
                    continue
                if a.get("acceptance"):
                    derived.append(str(a["acceptance"]))
                elif a.get("summary"):
                    derived.append(f"Implement: {a['summary']}")
            ac = derived if derived else None

        if not isinstance(ac, list):
            ac = [str(ac)] if ac is not None else []
        ac = [str(x).strip() for x in ac if str(x).strip()]
        if not ac:
            ac = ["Visible progress on this recommendation."]
        return ac

    def _normalize_single(
        item: Dict[str, Any],
        index_hint: int,
    ) -> Optional[Dict[str, Any]]:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        raw_reason = (item.get("reason") or item.get("why") or "").strip()
        # Use explicit rationale if present; otherwise we’ll derive one.
        raw_rationale = (item.get("rationale") or "").strip()
        raw_summary = (item.get("summary") or "").strip()

        ac = _ensure_acceptance(item)

        # Fill in reason / rationale / summary with sensible fallbacks.
        if not raw_reason and ac:
            raw_reason = "See acceptance items."
        if not raw_rationale:
            raw_rationale = raw_reason or "Grouped work item for this codebase."
        if not raw_summary:
            raw_summary = raw_reason or f"Implement the changes described for '{title}'."

        risk_val = _norm_risk(item.get("risk"))

        # IDs: preserve provided, otherwise synthesize stable-ish ID.
        id_val = (item.get("id") or "").strip()
        if not id_val:
            id_val = f"rec-{index_hint}"

        # Schema version: preserve if present/int; default to 2.
        sv = item.get("schema_version")
        try:
            schema_version = int(sv) if sv is not None else 2
        except Exception:
            schema_version = 2

        budget_estimate = item.get("budget_estimate")
        if not isinstance(budget_estimate, dict):
            budget_estimate = None

        files_val = _norm_files(item.get("files"))
        actions_val = _norm_actions(item.get("actions"))

        rec: Dict[str, Any] = {
            "schema_version": schema_version,
            "id": id_val,
            "title": title,
            "rationale": raw_rationale,
            "reason": raw_reason or "See acceptance items.",
            "summary": raw_summary,
            "risk": risk_val,
            "acceptance_criteria": ac,
        }
        if budget_estimate:
            rec["budget_estimate"] = budget_estimate
        if files_val:
            rec["files"] = files_val
        if actions_val:
            rec["actions"] = actions_val

        return rec

    # ---- Case 1: direct list of recommendation-like dicts ----
    if isinstance(payload, list):
        for idx, item in enumerate(payload, 1):
            if not isinstance(item, dict):
                continue
            rec = _normalize_single(item, idx)
            if rec:
                out.append(rec)
        return out

    # ---- Case 2: object-wrapped forms ----
    if isinstance(payload, dict):
        # { rationales: [...] } – legacy / alternate shape.
        rats = payload.get("rationales")
        if isinstance(rats, list):
            for idx, r in enumerate(rats, 1):
                if not isinstance(r, dict):
                    continue
                rec = _normalize_single(r, idx)
                if rec:
                    out.append(rec)

        # Fallback to nested `recommendations` or `items` if we didn't get anything yet.
        alt = payload.get("recommendations") or payload.get("items")
        if not out and isinstance(alt, list):
            out.extend(normalize_recommendations(alt))

    return out


def _strip_nulls(obj: Any) -> Any:
    """
    Recursively remove keys with None values from dictionaries and
    None items from lists. Leaves other falsy values (0, "", False)
    intact.

    This is used by the LLM client to clean payloads before sending
    them over the wire or persisting them.
    """
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj if v is not None]
    return obj
