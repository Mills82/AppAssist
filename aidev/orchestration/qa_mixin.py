# aidev/orchestration/qa_mixin.py
from __future__ import annotations

import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Tuple

from aidev.cards import KnowledgeBase
from aidev import events as _events  # kept for future event hooks if needed
from aidev import validators as _validators
from aidev.schemas import qa_answer_schema
from aidev.orchestration.qa_prompts import qa_system_prompt, build_qa_user_payload
# research_and_retry is an optional internal helper; import failure must not break module import
try:
    from aidev.orchestration.research import research_and_retry
except Exception:
    research_and_retry = None


def _extract_text_from(raw: Any) -> str:
    """Best-effort text extractor from raw model output for fallback use.

    This mirrors the local helper used by OrchestratorQAMixin._run_qa_pipeline so
    callers (including tests) can reuse the same extraction logic.
    """
    try:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            # prefer obvious fields
            for k in ("answer", "text", "content", "message"):
                v = raw.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # otherwise stringify small primitives or join string values
            parts = [str(v).strip() for v in raw.values() if isinstance(v, (str, int, float))]
            if parts:
                joined = " ".join(parts)
                return joined[:4000]
            # fallback to JSON
            return json.dumps(raw)[:4000]
        # fallback for other types
        return str(raw)[:4000]
    except Exception:
        return ""


def _safe_emit_qa_answer(payload: Dict[str, Any], question: str | None = None, answer: str | None = None, session_id: Any = None, job_id: Any = None) -> None:
    """Attempt to emit a qa_answer event using a few possible event signatures.

    This helper first tries the preferred _events.emit_qa_answer(event_type=..., payload=..., session_id=..., job_id=...).
    If that raises a TypeError (different signature), it will attempt a few fallbacks
    including _events.emit_qa_answer(payload=...), positional calls, and the legacy
    _events.qa_answer(...) helper. All failures are swallowed and logged at debug
    level to avoid crashing the QA pipeline.
    """
    try:
        if hasattr(_events, "emit_qa_answer"):
            try:
                _events.emit_qa_answer(event_type="qa_answer", payload=payload, session_id=session_id, job_id=job_id)
                return
            except TypeError:
                # try alternative keyword-only signature
                try:
                    _events.emit_qa_answer(payload=payload)
                    return
                except TypeError:
                    # try positional payload
                    try:
                        _events.emit_qa_answer(payload)
                        return
                    except Exception:
                        pass
        # fallback to legacy helper
        if hasattr(_events, "qa_answer"):
            try:
                if question is None and answer is None and isinstance(payload, dict):
                    answer = payload.get("answer")
                _events.qa_answer(question=question or "", answer=answer or "", session_id=session_id, job_id=job_id)
                return
            except TypeError:
                try:
                    _events.qa_answer(question or "", answer or "")
                    return
                except Exception:
                    pass
    except Exception:
        logging.debug("Failed to emit QA answer via helper", exc_info=True)


def enrich_file_refs(qa_obj: Dict[str, Any], kb: Any, root: str | None, max_len: int = 200) -> None:
    """
    Best-effort enrichment of qa_obj['file_refs'] entries by adding a short
    "snippet" (<= max_len chars) for each referenced file and normalizing
    entries to dicts with at least a 'path' key.

    Behavior and safety:
      - If a file_ref is a string it becomes {'path': <value>}.
      - If a file_ref is already a dict we shallow-copy it and preserve any
        numeric 'line_start'/'line_end' keys if present.
      - Snippet extraction preference order:
          1) If 'snippet' already present and non-empty, accept (truncate/normalize).
          2) If line_start/line_end present: try to extract those lines from the
             card (kb lookup) or from disk (under root).
          3) card.summary or card.text (via kb lookup) if available.
          4) Read the file from disk under root as a final fallback (best-effort,
             limited read).
      - Whitespace is collapsed (any \s+ -> single space) and snippets are
        trimmed and truncated to max_len.

    Safety:
      - Does not raise; any failure leaves the corresponding ref normalized
        minimally (dict with 'path').
      - When reading disk, realpath is used and files outside 'root' are not read.

    The function mutates qa_obj in-place. Designed to be deterministic and
    unit-test friendly.
    """
    try:
        if not qa_obj or not isinstance(qa_obj, dict):
            return
        refs = qa_obj.get("file_refs") or []
        if not isinstance(refs, list):
            return

        root_real = os.path.realpath(root) if root else None
        enriched: List[Dict[str, Any]] = []

        for ref in refs:
            try:
                # Normalize to a mutable dict with at least 'path'
                if isinstance(ref, str):
                    refd = {"path": ref}
                elif isinstance(ref, dict):
                    refd = dict(ref)
                else:
                    refd = {"path": str(ref)}

                path = refd.get("path")
                snippet = None

                # 1) accept existing snippet (normalize and truncate)
                if isinstance(refd.get("snippet"), str) and refd.get("snippet").strip():
                    s = re.sub(r"\s+", " ", refd.get("snippet")).strip()
                    if len(s) > max_len:
                        s = s[:max_len]
                    refd["snippet"] = s
                    enriched.append(refd)
                    continue

                # Helper: attempt to read limited file content from disk under root
                def _read_file_limited(pth: str, limit: int = 10000) -> str | None:
                    try:
                        if not root_real:
                            return None
                        candidate = os.path.join(root, pth)
                        cand_real = os.path.realpath(candidate)
                        if not cand_real.startswith(root_real):
                            return None
                        if not os.path.isfile(cand_real):
                            return None
                        with open(cand_real, "r", encoding="utf-8", errors="ignore") as fh:
                            return fh.read(limit)
                    except Exception:
                        return None

                # Try to find a card for the given path via common KB APIs
                card = None
                if hasattr(kb, "get_card"):
                    try:
                        card = kb.get_card(path)
                    except Exception:
                        card = None
                if card is None:
                    for fn in ("get_card_by_path", "card_for_path", "get", "find_card"):
                        if hasattr(kb, fn):
                            try:
                                card = getattr(kb, fn)(path)
                                break
                            except Exception:
                                card = None

                # If line ranges provided, prefer extracting those lines
                ls = refd.get("line_start")
                le = refd.get("line_end")
                if (ls is not None or le is not None):
                    content = None
                    # Prefer card text if available
                    if card is not None:
                        content = getattr(card, "text", None) or getattr(card, "summary", None)
                    if not content and path:
                        content = _read_file_limited(path)
                    if content:
                        try:
                            lines = content.splitlines()
                            start_idx = max(0, int(ls) - 1) if ls is not None else 0
                            end_idx = int(le) if le is not None else start_idx + 4
                            snippet_text = " ".join(lines[start_idx:end_idx])
                            if snippet_text:
                                snippet = snippet_text
                        except Exception:
                            snippet = None

                # If still no snippet, try card summary/text
                if snippet is None and card is not None:
                    candidate_text = getattr(card, "summary", None) or getattr(card, "text", None)
                    if isinstance(candidate_text, str) and candidate_text.strip():
                        snippet = candidate_text

                # Final fallback: read from disk (limited)
                if snippet is None and path:
                    content = _read_file_limited(path)
                    if content:
                        snippet = content

                # Normalize snippet if found
                if isinstance(snippet, str):
                    s = re.sub(r"\s+", " ", snippet).strip()
                    if len(s) > max_len:
                        s = s[:max_len]
                    refd["snippet"] = s

                # Ensure path is a string if present
                if path is not None:
                    try:
                        refd["path"] = str(path)
                    except Exception:
                        pass

                enriched.append(refd)
            except Exception:
                # per-ref failure: include minimal normalized record
                try:
                    if isinstance(ref, dict):
                        enriched.append(ref)
                    else:
                        enriched.append({"path": str(ref)})
                except Exception:
                    enriched.append({"path": ""})

        qa_obj["file_refs"] = enriched
    except Exception:
        # global failure: leave qa_obj unchanged (best-effort only)
        return


def validate_qa_answer(raw: Any, schema: Any | None = None) -> Tuple[Dict[str, Any], bool, Any]:
    """
    Validate and normalize a raw LLM QA response into the canonical QA object.

    Returns a tuple (normalized_obj, validation_ok, validation_errors).
    - normalized_obj: dict with at least 'answer', 'file_refs', 'follow_ups'
    - validation_ok: True if the object passed schema/structural validation
    - validation_errors: validator-specific errors or traceback on failure

    This function is provided at module level so unit tests can import and
    exercise the validation/fallback behavior independent of the orchestration
    class.
    """
    qa_schema = schema or qa_answer_schema()

    qa_obj: Dict[str, Any] = {"answer": "", "file_refs": [], "follow_ups": []}
    raw_data = raw

    try:
        if isinstance(raw, dict):
            qa_obj["answer"] = str(raw.get("answer", "")).strip()
            # Map legacy key 'files' to the new 'file_refs' to maintain
            # backward compatibility with older model outputs.
            if "file_refs" in raw:
                qa_obj["file_refs"] = list(raw.get("file_refs") or [])
            elif "files" in raw:
                qa_obj["file_refs"] = list(raw.get("files") or [])
            else:
                qa_obj["file_refs"] = list(raw.get("file_refs") or [])
            qa_obj["follow_ups"] = list(raw.get("follow_ups") or [])
            # keep other fields if present under a meta key
            meta_fields = {k: v for k, v in raw.items() if k not in qa_obj}
            if meta_fields:
                qa_obj["meta"] = meta_fields
        elif isinstance(raw, str):
            # try to parse JSON strings
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    qa_obj["answer"] = str(parsed.get("answer", "")).strip()
                    if "file_refs" in parsed:
                        qa_obj["file_refs"] = list(parsed.get("file_refs") or [])
                    elif "files" in parsed:
                        qa_obj["file_refs"] = list(parsed.get("files") or [])
                    else:
                        qa_obj["file_refs"] = list(parsed.get("file_refs") or [])
                    qa_obj["follow_ups"] = list(parsed.get("follow_ups") or [])
                    meta_fields = {k: v for k, v in parsed.items() if k not in qa_obj}
                    if meta_fields:
                        qa_obj["meta"] = meta_fields
                else:
                    # model returned JSON but not an object; treat as text
                    qa_obj["answer"] = raw.strip()
            except Exception:
                # plain text answer
                qa_obj["answer"] = raw.strip()
        else:
            qa_obj["answer"] = str(raw).strip()
    except Exception:
        # Parsing failure: build a fallback object using the extractor.
        qa_obj = {
            "answer": _extract_text_from(raw_data),
            "file_refs": [],
            "follow_ups": [],
        }

    # Validate the normalized object using the validators helper when
    # available. Accept multiple validator return styles gracefully.
    validation_ok = False
    validation_errors = None
    try:
        if hasattr(_validators, "validate_schema"):
            res = _validators.validate_schema(qa_schema, qa_obj)
            # interpret common return values
            if res in (None, True):
                validation_ok = True
            elif isinstance(res, tuple) and len(res) == 2:
                validation_ok, validation_errors = res
            elif isinstance(res, (list, dict)):
                # treat empty list/dict as OK
                if not res:
                    validation_ok = True
                else:
                    validation_errors = res
            else:
                # truthy non-standard value -> consider ok
                validation_ok = bool(res)
        elif hasattr(_validators, "validate"):
            res = _validators.validate(qa_schema, qa_obj)
            if isinstance(res, tuple) and len(res) == 2:
                validation_ok, validation_errors = res
            elif res in (None, True):
                validation_ok = True
            elif isinstance(res, (list, dict)):
                if not res:
                    validation_ok = True
                else:
                    validation_errors = res
            else:
                validation_ok = bool(res)
        else:
            # No validators available: fall back to minimal structural checks
            if isinstance(qa_obj.get("answer"), str) and isinstance(qa_obj.get("file_refs"), list) and isinstance(qa_obj.get("follow_ups"), list):
                validation_ok = True
            else:
                validation_errors = "minimal-schema-check-failed"
    except Exception:
        validation_errors = traceback.format_exc()

    if not validation_ok:
        # Build fallback normalized object
        fallback_qa_obj = {
            "answer": qa_obj.get("answer", "") or _extract_text_from(raw_data),
            "file_refs": [],
            "follow_ups": [],
        }
        if qa_obj.get("meta"):
            fallback_qa_obj["meta"] = qa_obj.get("meta")
        return fallback_qa_obj, False, validation_errors

    return qa_obj, True, validation_errors


class OrchestratorQAMixin:
    """
    Mixin that holds the conversational Q&A mode logic so aidev/orchestrator.py
    stays lean.

    This assumes `self` is an Orchestrator-like object with:

      - args, st, root, job_id, _session_id
      - _llm, _project_brief_text
      - methods: _should_cancel, _progress, _progress_error, _job_update,
                 _chat_json, _to_jsonable, _phase_max_tokens,
                 _emit_qa_answer

    NOTE: This QA pipeline emits a server-sent event with event_type 'qa_answer'.
    The payload is validated/normalized to match the schema returned by
    aidev/schemas/qa_answer.schema.json (via qa_answer_schema()) and should be
    an object with at least the keys: 'answer' (string), 'file_refs' (list), and
    'follow_ups' (list). Validators are used when available; parse/validation
    failures result in a structured error event being emitted but the pipeline
    will continue and emit a minimal, schema-conformant fallback object so
    downstream consumers never receive an exception or missing payload.
    """

    def _run_qa_pipeline(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        ctx_blob: Any,
    ) -> Tuple[bool, str]:
        """
        Conversational Q&A mode: answer questions about the project without
        generating structured recommendations or edits.

        This is optimized for:
          - minimal, well-bounded context size
          - stable JSON output for the UI
          - clear, human-friendly answers
        """
        if self._should_cancel():
            return False, "Run cancelled before Q&A."

        # --------- resolve the question text ---------
        question = ""
        try:
            # Prefer explicit `question`, fall back to `focus` or `prompt`.
            raw = (
                self.args.get("question")
                or self.args.get("focus")
                or self.args.get("prompt")
                or ""
            )
            question = str(raw).strip()
        except Exception:
            question = ""

        if not question:
            self._progress_error(
                "qa_mode",
                reason="no_question",
                message="QA mode requested but no question/focus provided.",
            )
            return False, "QA mode: no question provided."

        # --------- select a small, high-signal set of cards ---------
        cfg = self.args.get("cfg") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        cards_cfg = cfg.get("cards", {}) if isinstance(cfg.get("cards"), dict) else {}

        try:
            top_k_default = int(cards_cfg.get("default_top_k", 5))
        except Exception:
            top_k_default = 5

        # For Q&A we cap to a small number to keep context tiny.
        try:
            top_k = int(self.args.get("cards_top_k") or top_k_default)
        except Exception:
            top_k = top_k_default

        if top_k <= 0:
            top_k = top_k_default

        try:
            hits = kb.select_cards(question, top_k=top_k)
        except Exception as e:
            hits = []
            self._progress_error(
                "qa_mode_select_cards",
                error=str(e),
                trace=traceback.format_exc(),
            )

        cards_payload: List[Dict[str, Any]] = []

        # Accept multiple select_cards return shapes for backwards/forwards compatibility:
        # - list of (card_obj, score) tuples
        # - list of dicts with keys: 'id','path','summary','score','title','language'
        # - list of card-like objects (with attributes)
        for hit in hits or []:
            try:
                # dict-shaped hit (newer API)
                if isinstance(hit, dict):
                    path = hit.get("path")
                    title = hit.get("title")
                    summary = hit.get("summary")
                    language = hit.get("language")
                    score = hit.get("score", 0)
                else:
                    # try tuple (card_obj, score)
                    card_obj = None
                    score = 0
                    try:
                        # may be (card, score)
                        card_obj, score = hit
                    except Exception:
                        # not a tuple, treat as a card-like object
                        card_obj = hit
                        try:
                            score = float(getattr(hit, "score", 0) or 0)
                        except Exception:
                            score = 0
                    # extract fields from card_obj if possible
                    if isinstance(card_obj, dict):
                        path = card_obj.get("path")
                        title = card_obj.get("title")
                        summary = card_obj.get("summary")
                        language = card_obj.get("language")
                    else:
                        path = getattr(card_obj, "path", None)
                        title = getattr(card_obj, "title", None)
                        summary = getattr(card_obj, "summary", None)
                        language = getattr(card_obj, "language", None)

                if isinstance(summary, str) and len(summary) > 800:
                    summary = summary[:800] + "…"

                cards_payload.append(
                    {
                        "path": path,
                        "title": title,
                        "summary": summary,
                        "language": language,
                        "score": float(score or 0),
                    }
                )
            except Exception:
                continue

        # Track whether we've retried using the internal research helper so we
        # only perform one additional LLM call at most.
        did_retry = False
        original_hits_empty = not bool(hits)

        # --------- build system prompt + user payload ---------
        system_text = qa_system_prompt()

        user_payload = build_qa_user_payload(
            question=question,
            project_brief=self._project_brief_text or "",
            project_meta=self._to_jsonable(meta),
            structure_overview=self._to_jsonable(ctx_blob),
            top_cards=cards_payload,
        )

        qa_schema = qa_answer_schema()

        self._job_update(
            stage="qa",
            message="Answering question about the project…",
            progress_pct=15,
        )
        self._progress("qa_mode_start", question=question)

        # --------- call LLM via the json-schema path with optional one-time retry ---------
        try:
            max_tokens = self._phase_max_tokens("qa")
        except Exception:
            max_tokens = None

        # Helper to run a single LLM attempt and return structured outcomes
        def _run_single_attempt(payload: Dict[str, Any]):
            """Returns (qa_obj, validation_ok, validation_errors, parse_failed, raw_data)
            qa_obj is the normalized or fallback object; parse_failed True means parsing error occurred.
            """
            parse_failed = False
            raw_data = None
            qa_obj_local: Dict[str, Any] = {"answer": "", "file_refs": [], "follow_ups": []}
            try:
                data, _res = self._chat_json(
                    system_text,
                    payload,
                    schema=qa_schema,
                    temperature=0.0,
                    phase="qa",
                    inject_brief=False,
                    max_tokens=max_tokens,
                )
                raw_data = data

                # Normalize/parse the model output into canonical QA object
                if isinstance(data, dict):
                    qa_obj_local["answer"] = str(data.get("answer", "")).strip()
                    if "file_refs" in data:
                        qa_obj_local["file_refs"] = list(data.get("file_refs") or [])
                    elif "files" in data:
                        qa_obj_local["file_refs"] = list(data.get("files") or [])
                    else:
                        qa_obj_local["file_refs"] = list(data.get("file_refs") or [])
                    qa_obj_local["follow_ups"] = list(data.get("follow_ups") or [])
                    meta_fields = {k: v for k, v in data.items() if k not in qa_obj_local}
                    if meta_fields:
                        qa_obj_local["meta"] = meta_fields
                elif isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, dict):
                            qa_obj_local["answer"] = str(parsed.get("answer", "")).strip()
                            if "file_refs" in parsed:
                                qa_obj_local["file_refs"] = list(parsed.get("file_refs") or [])
                            elif "files" in parsed:
                                qa_obj_local["file_refs"] = list(parsed.get("files") or [])
                            else:
                                qa_obj_local["file_refs"] = list(parsed.get("file_refs") or [])
                            qa_obj_local["follow_ups"] = list(parsed.get("follow_ups") or [])
                            meta_fields = {k: v for k, v in parsed.items() if k not in qa_obj_local}
                            if meta_fields:
                                qa_obj_local["meta"] = meta_fields
                        else:
                            qa_obj_local["answer"] = data.strip()
                    except Exception:
                        qa_obj_local["answer"] = data.strip()
                else:
                    qa_obj_local["answer"] = str(data).strip()
            except Exception:
                # Parsing failure or chat_json failure: build fallback but mark parse_failed
                parse_failed = True
                raw_data = None
                try:
                    # if there was an exception, try to capture traceback as raw_data for observability
                    raw_data = traceback.format_exc()
                except Exception:
                    raw_data = None
                qa_obj_local = {
                    "answer": _extract_text_from(raw_data),
                    "file_refs": [],
                    "follow_ups": [],
                }

            # Validate the normalized object
            validation_ok = False
            validation_errors = None
            try:
                if hasattr(_validators, "validate_schema"):
                    res = _validators.validate_schema(qa_schema, qa_obj_local)
                    if res in (None, True):
                        validation_ok = True
                    elif isinstance(res, tuple) and len(res) == 2:
                        validation_ok, validation_errors = res
                    elif isinstance(res, (list, dict)):
                        if not res:
                            validation_ok = True
                        else:
                            validation_errors = res
                    else:
                        validation_ok = bool(res)
                elif hasattr(_validators, "validate"):
                    res = _validators.validate(qa_schema, qa_obj_local)
                    if isinstance(res, tuple) and len(res) == 2:
                        validation_ok, validation_errors = res
                    elif res in (None, True):
                        validation_ok = True
                    elif isinstance(res, (list, dict)):
                        if not res:
                            validation_ok = True
                        else:
                            validation_errors = res
                    else:
                        validation_ok = bool(res)
                else:
                    if isinstance(qa_obj_local.get("answer"), str) and isinstance(qa_obj_local.get("file_refs"), list) and isinstance(qa_obj_local.get("follow_ups"), list):
                        validation_ok = True
                    else:
                        validation_errors = "minimal-schema-check-failed"
            except Exception:
                validation_errors = traceback.format_exc()

            return qa_obj_local, validation_ok, validation_errors, parse_failed, raw_data

        # First attempt
        try:
            qa_obj, validation_ok, validation_errors, parse_failed, raw_data = _run_single_attempt(user_payload)
        except Exception as e:
            # If something unexpected happens, record and fail gracefully
            self._progress_error(
                "qa_mode_chat",
                error=str(e),
                trace=traceback.format_exc(),
            )
            return False, "QA mode failed while calling LLM."

        # Determine confidence if present
        low_confidence = False
        try:
            meta_conf = None
            if isinstance(qa_obj, dict):
                meta_conf = qa_obj.get("meta", {}) and qa_obj.get("meta", {}).get("confidence")
            if meta_conf is not None:
                try:
                    conf_val = float(meta_conf)
                    if conf_val < 0.6:
                        low_confidence = True
                except Exception:
                    # non-numeric confidence -> ignore
                    pass
        except Exception:
            pass

        answer_empty = not bool(str(qa_obj.get("answer", "") or "").strip())

        # Decide whether to invoke the one-time internal research fallback and retry
        need_retry = False
        if (original_hits_empty or parse_failed or (not validation_ok) or low_confidence or answer_empty) and (not did_retry):
            need_retry = True

        if need_retry and research_and_retry is not None:
            try:
                # Observability: record that we're attempting research and retry
                self._progress("qa_research_retry", question=question)
                research_result = research_and_retry(question, self, hits)
                # research_and_retry expected to return (enriched_payload, did_perform)
                if isinstance(research_result, tuple) and len(research_result) >= 1:
                    enriched_payload = research_result[0]
                    performed = bool(research_result[1]) if len(research_result) > 1 else bool(research_result[0])
                else:
                    enriched_payload = research_result
                    performed = bool(research_result)
                if enriched_payload and performed:
                    did_retry = True
                    user_payload = enriched_payload
                    # Re-run LLM once with enriched payload
                    try:
                        qa_obj, validation_ok, validation_errors, parse_failed, raw_data = _run_single_attempt(user_payload)
                    except Exception:
                        # If retry fails catastrophically, fall through to emission of fallback
                        qa_obj = {"answer": _extract_text_from(raw_data), "file_refs": [], "follow_ups": []}
                        validation_ok = False
                        parse_failed = True
                # else: research helper didn't perform enrichment; continue with previous qa_obj
            except Exception:
                # research helper failure must not crash; log and continue
                logging.debug("research_and_retry failed", exc_info=True)

        # After attempts: surface parse/validation errors as structured events for observability
        try:
            if parse_failed:
                error_payload = {
                    "error": "parse_failed",
                    "details": str(validation_errors) if validation_errors else "parse_failed",
                    "raw": raw_data,
                }
                try:
                    _safe_emit_qa_answer(error_payload, question=question, answer="", session_id=getattr(self, "_session_id", None), job_id=getattr(self, "job_id", None))
                except Exception:
                    logging.debug("Failed to emit QA parse error event", exc_info=True)
                try:
                    self.st.trace.write("QA", "error", error_payload)
                except Exception:
                    logging.debug("Failed to write QA parse error to trace", exc_info=True)

            if not validation_ok:
                error_payload = {
                    "error": "validation_failed",
                    "details": validation_errors,
                    "raw": raw_data,
                }
                try:
                    _safe_emit_qa_answer(error_payload, question=question, answer=qa_obj.get("answer", ""), session_id=getattr(self, "_session_id", None), job_id=getattr(self, "job_id", None))
                except Exception:
                    logging.debug("Failed to emit QA validation error event", exc_info=True)
                try:
                    self.st.trace.write("QA", "error", error_payload)
                except Exception:
                    logging.debug("Failed to write QA validation error to trace", exc_info=True)

                # Build fallback for downstream consumers
                fallback_qa_obj = {
                    "answer": qa_obj.get("answer", "") or _extract_text_from(raw_data),
                    "file_refs": [],
                    "follow_ups": [],
                }
                if qa_obj.get("meta"):
                    fallback_qa_obj["meta"] = qa_obj.get("meta")
                try:
                    try:
                        enrich_file_refs(fallback_qa_obj, kb, getattr(self, "root", None))
                    except Exception:
                        pass
                    _safe_emit_qa_answer(fallback_qa_obj, question=question, answer=fallback_qa_obj.get("answer", ""), session_id=getattr(self, "_session_id", None), job_id=getattr(self, "job_id", None))
                except Exception:
                    logging.debug("Failed to emit fallback QA validation object", exc_info=True)
                try:
                    self.st.trace.write("QA", "answer", fallback_qa_obj)
                except Exception:
                    logging.debug("Failed to write fallback QA answer to trace", exc_info=True)

                # Continue: do not return early; also emit canonical/legacy events below
                qa_obj = fallback_qa_obj

            # Success (or fallback): emit the validated/fallback QA object as a named SSE event so UI clients receive stable, schema-compliant payloads.
            try:
                try:
                    enrich_file_refs(qa_obj, kb, getattr(self, "root", None))
                except Exception:
                    pass

                _safe_emit_qa_answer(qa_obj, question=question, answer=qa_obj.get("answer", ""), session_id=getattr(self, "_session_id", None), job_id=getattr(self, "job_id", None))
            except Exception:
                logging.debug("Failed to emit QA answer event", exc_info=True)

            try:
                # Record the validated qa_obj in the run trace
                self.st.trace.write("QA", "answer", qa_obj)
            except Exception:
                logging.debug("Failed to write validated QA answer to trace", exc_info=True)

            # Provide convenience local variables for the rest of the pipeline
            answer = qa_obj.get("answer", "")
            extra: Dict[str, Any] = {k: v for k, v in qa_obj.items() if k != "answer"}

        except Exception as e:
            self._progress_error(
                "qa_mode_chat",
                error=str(e),
                trace=traceback.format_exc(),
            )
            return False, "QA mode failed while processing LLM result."

        # --------- emit events + trace (legacy/secondary) ---------
        try:
            try:
                _events.qa_answer(
                    question=question,
                    answer=answer,
                    session_id=getattr(self, "_session_id", None),
                    job_id=getattr(self, "job_id", None),
                )
            except Exception:
                logging.debug("Legacy _events.qa_answer emission failed", exc_info=True)

            payload: Dict[str, Any] = {"question": question, "answer": answer}
            if extra:
                payload["meta"] = extra

            try:
                self.st.trace.write("QA", "answer", payload)
            except Exception:
                logging.debug("Failed to write QA payload to trace", exc_info=True)
        except Exception:
            logging.debug("Failed to emit QA answer/trace", exc_info=True)

        self._progress("qa_mode_done", question=question)
        self._job_update(
            stage="qa",
            message="Q&A completed.",
            progress_pct=100,
        )

        return True, "Q&A completed."
