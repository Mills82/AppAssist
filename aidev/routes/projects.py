# aidev/routes/projects.py
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from jsonschema import validate as jsonschema_validate, ValidationError as JSONSchemaValidationError

# Starlette cookie session helper
from aidev.routes.session import get_session_store  # type: ignore

# Bridge to the SSE-backed SessionStore used by /workspaces/*
from ..session_store import SESSIONS
from ..events import project_selected as ev_project_selected

from ..context.brief import get_or_build, compute_brief_hash  # type: ignore

# Optional runtime detector
try:
    from runtimes import detect_runtimes  # type: ignore
except Exception:  # pragma: no cover
    detect_runtimes = None

# Try importing the conversational project create flow and session helpers.
try:
    # Prefer importing the flow and session helpers from the stages module so
    # session persistence is shared across the codebase.
    from aidev.stages.project_create_flow import (
        create_project_flow,
        save_session as save_session_external,
        load_session as load_session_external,
        delete_session as delete_session_external,
    )  # type: ignore
except Exception:  # pragma: no cover
    create_project_flow = None
    save_session_external = None
    load_session_external = None
    delete_session_external = None

try:
    from aidev.io_utils import atomic_write  # type: ignore
except Exception:
    atomic_write = None

# ✅ OpenAI-only, Responses-only LLM client
try:
    from ..llm_client import ChatGPT  # alias for LLMClient
except Exception:
    ChatGPT = None  # type: ignore

router = APIRouter(prefix="/projects", tags=["projects"])


# ---------- Utilities

def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_\.]+", "-", name.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s or f"proj-{uuid.uuid4().hex[:8]}"


def _schema_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "schemas"


def _load_schema(name: str) -> Dict[str, Any]:
    p = _schema_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"Schema not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_payload(payload: Dict[str, Any], schema_file: str) -> Dict[str, Any]:
    schema = _load_schema(schema_file)
    try:
        jsonschema_validate(payload, schema)
        return payload
    except JSONSchemaValidationError as e:
        raise HTTPException(status_code=422, detail=f"Schema validation failed: {e.message}") from e


def _read_prompt(name: str) -> str:
    p = Path(__file__).resolve().parent.parent / "prompts" / name
    if not p.exists():
        raise HTTPException(status_code=500, detail=f"Prompt missing: {name}")
    return p.read_text(encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _unique_dir(base: Path, slug: str) -> Path:
    candidate = base / slug
    if not candidate.exists():
        return candidate
    return base / f"{slug}-{uuid.uuid4().hex[:6]}"


def _unified_diff(old_text: str, new_text: str, a: str, b: str) -> str:
    import difflib
    lines = list(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=a,
            tofile=b,
            lineterm="",
        )
    )
    return "".join(lines)


def _stable_project_id(p: Path) -> str:
    # UUID v5 from canonicalized path (matches workspaces’ approach)
    key = p.expanduser().resolve().as_posix().lower()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"file://{key}"))


def _read_project_descriptions(root: Path):
    """
    Read human-written and compiled descriptions for a project.

    Returns:
        (app_description, compiled_md)
    """
    app_description = ""
    compiled_md = ""
    try:
        # Human source-of-truth: .aidev/app_descrip.txt, with root/app_descrip.txt fallback
        candidates_app = [
            root / ".aidev" / "app_descrip.txt",
            root / "app_descrip.txt",
        ]
        for p in candidates_app:
            if p.exists():
                app_description = p.read_text(encoding="utf-8").strip()
                break

        # LLM-compiled markdown: .aidev/project_description.md, with root/project_description.md fallback
        candidates_compiled = [
            root / ".aidev" / "project_description.md",
            root / "project_description.md",
        ]
        for p in candidates_compiled:
            if p.exists():
                compiled_md = p.read_text(encoding="utf-8").strip()
                break
    except Exception:
        # Non-fatal; just return empty strings
        pass

    return app_description, compiled_md


# ---------- LLM helpers (OpenAI-only, Responses-only)

def _require_llm() -> Any:
    if ChatGPT is None:
        raise HTTPException(status_code=500, detail="LLM client unavailable: failed to import aidev.llm_client.ChatGPT")
    # Ensure key exists early so failures are clear
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    return ChatGPT


def _llm_json(system_prompt: str, user_payload: Any, *, phase: str, stage: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic JSON call. Uses Responses-only LLMClient.chat_json().
    """
    LLM = _require_llm()
    client = LLM()
    try:
        user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
        data, _resp = client.chat_json(
            [{"role": "user", "content": user_text}],
            schema=None,  # schema-less: rely on strict JSON parsing + downstream normalization
            system=system_prompt,
            max_tokens=None,
            extra={"phase": phase, "disable_web_search": True},
            stage=stage or phase,
        )
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail=f"LLM returned non-object JSON for {phase}: {type(data).__name__}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error ({phase}): {e}") from e
    finally:
        try:
            client.close()
        except Exception:
            pass


# ---------- Request models

class CreateProjectBody(BaseModel):
    brief: str
    project_name: Optional[str] = None
    base_dir: Optional[str] = None  # defaults to CWD/projects
    # New fields to support conversational follow-ups / resumption
    session_token: Optional[str] = None
    answers: Optional[Dict[str, Any]] = None


class SelectProjectBody(BaseModel):
    project_path: str


class UpdateDescriptionsBody(BaseModel):
    app_description: str


class RunChecksBody(BaseModel):
    project_path: Optional[str] = None


# ---------- Small helpers for session persistence & atomic writes

def _session_dir_for_project(aidev_dir: Path) -> Path:
    d = aidev_dir / "project_create_sessions"
    _ensure_dir(d)
    return d


def _save_session_local(aidev_dir: Path, token: str, envelope: Dict[str, Any]) -> None:
    sd = _session_dir_for_project(aidev_dir)
    p = sd / f"{token}.json"
    p.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def _load_session_local(aidev_dir: Path, token: str) -> Optional[Dict[str, Any]]:
    sd = _session_dir_for_project(aidev_dir)
    p = sd / f"{token}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _delete_session_local(aidev_dir: Path, token: str) -> None:
    sd = _session_dir_for_project(aidev_dir)
    p = sd / f"{token}.json"
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _default_atomic_write(target: Path, content: str, encoding: str = "utf-8") -> None:
    # Simple atomic write fallback: write to temp file in same dir then replace
    target_parent = target.parent
    _ensure_dir(target_parent)
    tmp = target_parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    tmp.write_text(content, encoding=encoding)
    os.replace(str(tmp), str(target))


def _write_file_atomic(target: Path, content: str) -> None:
    if atomic_write is not None:
        try:
            atomic_write(target, content)
            return
        except Exception:
            # fallback to default
            pass
    _default_atomic_write(target, content)


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        base_res = base.resolve()
        target_res = target.resolve()
        # On Python >=3.9 we could use is_relative_to; fallback to commonpath
        return os.path.commonpath([str(base_res)]) == os.path.commonpath([str(base_res), str(target_res)])
    except Exception:
        return False


# Wrapper helpers that prefer the external session helpers when available.
def save_session_shared(aidev_dir: Path, token: str, envelope: Dict[str, Any]) -> None:
    """Persist a transient flow session. If the stages module exposes save_session(token, envelope)
    use that; otherwise fall back to writing under project .aidev/project_create_sessions/.
    """
    if save_session_external:
        # external helper likely manages storage location itself and expects (token, envelope)
        try:
            save_session_external(token, envelope)
            return
        except TypeError:
            # some historical variants may accept (aidev_dir, token, envelope)
            try:
                save_session_external(aidev_dir, token, envelope)
                return
            except Exception:
                pass
        except Exception:
            pass
    # fallback
    _save_session_local(aidev_dir, token, envelope)


def load_session_shared(aidev_dir: Path, token: str) -> Optional[Dict[str, Any]]:
    """Load a previously persisted transient flow session.
    Prefer external loader if available.
    """
    if load_session_external:
        try:
            res = load_session_external(token)
            return res
        except TypeError:
            try:
                return load_session_external(aidev_dir, token)
            except Exception:
                pass
        except Exception:
            pass
    return _load_session_local(aidev_dir, token)


def delete_session_shared(aidev_dir: Path, token: str) -> None:
    """Delete a persisted transient flow session. Prefer external helper when available."""
    if delete_session_external:
        try:
            delete_session_external(token)
            return
        except TypeError:
            try:
                delete_session_external(aidev_dir, token)
                return
            except Exception:
                pass
        except Exception:
            pass
    _delete_session_local(aidev_dir, token)


def _normalize_follow_up_questions(items: Any) -> list:
    """Normalize follow_up_questions into canonical objects:
    {id: str, question: str, type: 'text'|'choice', required: bool, choices?: list}
    Accept strings (converted to required text questions) and dicts; raise HTTPException on malformed items.
    """
    if not isinstance(items, list):
        raise HTTPException(status_code=502, detail="follow_up_questions must be an array")
    normalized = []
    for it in items:
        if isinstance(it, str):
            normalized.append({"id": uuid.uuid4().hex, "question": it, "type": "text", "required": True})
            continue
        if not isinstance(it, dict):
            raise HTTPException(status_code=502, detail="Malformed follow_up_questions item; must be string or object")
        qid = it.get("id") or it.get("name") or uuid.uuid4().hex
        question = it.get("question") or it.get("text")
        if not question or not isinstance(question, str):
            raise HTTPException(status_code=502, detail="Malformed follow_up_questions item missing 'question' string")
        qtype = str(it.get("type", "text"))
        if qtype not in ("text", "choice"):
            qtype = "text"
        required = bool(it.get("required", False))
        entry = {"id": str(qid), "question": question, "type": qtype, "required": required}
        if qtype == "choice":
            choices = it.get("choices") or it.get("options") or []
            if not isinstance(choices, list):
                raise HTTPException(status_code=502, detail="Malformed follow_up_questions 'choices' must be array")
            entry["choices"] = choices
        normalized.append(entry)
    return normalized


# ---------- Endpoints

@router.post("/create")
async def create_project(
    req: Request,
    body: CreateProjectBody,
    session_id: Optional[str] = Query(default=None, description="Bridge to SSE SessionStore"),
) -> Dict[str, Any]:
    """
    Create a project folder using LLM or the project_create_flow, persist files, and set active project in both:
      1) Starlette cookie session (request.session)
      2) SSE-backed SessionStore (if session_id provided)

    The flow may return follow_up_questions (in which case a transient session_token is returned and files are not written),
    or it may return created_files immediately (which will be validated and written atomically).
    """
    payload = _validate_payload(body.model_dump(), "project_create_request.schema.json")

    cookie_session = get_session_store(req)  # Starlette cookie session

    brief: str = payload.get("brief") or ""
    if not brief.strip() and not payload.get("session_token"):
        raise HTTPException(status_code=400, detail="brief is required when starting a new project flow")

    project_name: str = payload.get("project_name") or (brief.splitlines()[0][:60] if brief else "new-project")
    base_dir = Path(payload.get("base_dir") or (Path.cwd() / "projects")).resolve()

    _ensure_dir(base_dir)
    slug = _slugify(project_name)
    project_dir = _unique_dir(base_dir, slug)
    _ensure_dir(project_dir)

    # Ensure .aidev exists for description files and session storage
    aidev_dir = project_dir / ".aidev"
    _ensure_dir(aidev_dir)

    # Decide whether to call the external project_create_flow or fall back to the legacy LLM flow
    answers = payload.get("answers") or None
    provided_token = payload.get("session_token") or None

    # If a session_token was provided, attempt to load envelope (to support resuming)
    loaded = None
    if provided_token:
        try:
            loaded = load_session_shared(aidev_dir, provided_token)
            if loaded:
                # allow loaded envelope to provide missing brief or other state
                brief = brief or loaded.get("brief") or brief
        except Exception:
            # If load fails treat as no session (non-fatal)
            loaded = None

    # Try calling create_project_flow when available
    flow_result = None
    try:
        if create_project_flow is not None:
            # Try a few possible call signatures to remain compatible with different versions.
            # Preferred minimal: brief + answers (+ flow_state if resuming).
            tried = False
            exc = None
            try:
                if loaded and "flow_state" in loaded:
                    flow_result = create_project_flow(brief=brief, answers=answers, flow_state=loaded.get("flow_state"))
                else:
                    flow_result = create_project_flow(brief=brief, answers=answers)
                tried = True
            except TypeError as e:
                exc = e

            if not tried:
                try:
                    # Some versions expect project_root path
                    flow_result = create_project_flow(brief=brief, project_root=str(project_dir), answers=answers)
                    tried = True
                except TypeError:
                    pass

            if not tried:
                # Some versions expect an llm_client object; provide a thin adapter implementing chat_json(prompt_path, payload)
                try:
                    class _LLMAdapter:
                        """
                        Adapter expected by some legacy project_create_flow implementations.

                        This is OpenAI-only, Responses-only.
                        chat_json(prompt_path, payload) -> dict
                        """
                        def chat_json(self, prompt_path, payload):
                            # prompt_path is usually a filename under prompts/, e.g. "system.project_create.md"
                            system_prompt = _read_prompt(str(prompt_path))
                            return _llm_json(
                                system_prompt=system_prompt,
                                user_payload=payload,
                                phase="project_create_flow",
                                stage="project_create",
                            )

                    llm_adapter = _LLMAdapter()
                    if loaded and "flow_state" in loaded:
                        flow_result = create_project_flow(
                            brief=brief,
                            llm_client=llm_adapter,
                            answers=answers,
                            flow_state=loaded.get("flow_state"),
                        )
                    else:
                        flow_result = create_project_flow(brief=brief, llm_client=llm_adapter, answers=answers)
                    tried = True
                except Exception:
                    # Fall-through; we'll raise below if nothing worked
                    pass

            if not tried and exc:
                raise exc

        else:
            # Fallback: legacy single-shot LLM flow (Responses-only).
            system_prompt = _read_prompt("system.project_create.md")
            parsed = _llm_json(
                system_prompt=system_prompt,
                user_payload=brief.strip(),
                phase="project_create_legacy",
                stage="project_create",
            )

            # Normalize to the newer shape expected by this route
            files = parsed.get("files", {}) if isinstance(parsed, dict) else {}
            created_files = []
            if isinstance(files, dict):
                if files.get("app_descrip.txt"):
                    created_files.append({"path": ".aidev/app_descrip.txt", "content": files.get("app_descrip.txt")})
                if files.get("project_description.md"):
                    created_files.append(
                        {"path": ".aidev/project_description.md", "content": files.get("project_description.md")}
                    )
            flow_result = {"created_files": created_files, "meta": parsed.get("meta") or {"title": parsed.get("title")}}

    except HTTPException:
        raise
    except Exception as e:
        # Flow/LLM errors are surfaced as 502 per existing semantics
        raise HTTPException(status_code=502, detail=f"project_create_flow failed: {e}") from e

    if not isinstance(flow_result, dict):
        raise HTTPException(status_code=502, detail="project_create_flow returned unexpected response")

    # If the flow requests follow-up questions, persist a transient session envelope and return questions + token
    if flow_result.get("follow_up_questions"):
        raw_questions = flow_result["follow_up_questions"]
        # Normalize and validate the questions so UI can render them deterministically.
        questions = _normalize_follow_up_questions(raw_questions)
        token = provided_token or uuid.uuid4().hex
        envelope = {
            "brief": brief,
            "project_path": str(project_dir),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "flow_state": flow_result.get("flow_state") or {},
            "questions": questions,
        }
        try:
            # Prefer shared session helper (may be implemented by stages.create flow)
            save_session_shared(aidev_dir, token, envelope)
        except Exception:
            # Non-fatal: still return the token, but warn in response
            return {
                "ok": True,
                "follow_up_questions": questions,
                "session_token": token,
                "warning": "failed to persist session",
            }
        return {"ok": True, "follow_up_questions": questions, "session_token": token}

    # Otherwise, expect created_files and write them atomically into project_dir
    created_files = flow_result.get("created_files") or []
    if not isinstance(created_files, list):
        raise HTTPException(status_code=502, detail="project_create_flow returned invalid created_files")

    files_written = []
    written_entries = []
    try:
        for entry in created_files:
            if not isinstance(entry, dict) or "path" not in entry or "content" not in entry:
                raise HTTPException(status_code=400, detail="created_files must be list of {path, content} dicts")
            rel_path = entry["path"]
            # Normalize relative paths: allow leading ./, but forbid absolute or traversal outside project_dir
            target = (project_dir / rel_path).resolve()
            if not _is_within_directory(project_dir, target):
                raise HTTPException(status_code=400, detail=f"Attempt to write outside project root: {rel_path}")
            # Ensure parent dirs exist
            _ensure_dir(target.parent)
            # Atomically write content
            _write_file_atomic(target, entry["content"])
            files_written.append(str(target.relative_to(project_dir)))
            written_entries.append({"path": str(target.relative_to(project_dir)), "content": entry["content"]})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write created files: {e}") from e

    # Compose project metadata and persist .aidev.project.json (preserve existing behaviour)
    project_id = _stable_project_id(project_dir)
    created_at = datetime.utcnow().isoformat() + "Z"
    meta = {
        "project_id": project_id,
        "name": project_name,
        "slug": slug,
        "created_at": created_at,
        "path": str(project_dir),
        # merge any meta from flow_result
        "llm_summary": flow_result.get("meta") or flow_result.get("llm_summary") or {},
    }
    try:
        (project_dir / ".aidev.project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        # Non-fatal
        pass

    # 1) Cookie session
    cookie_session["project_path"] = str(project_dir)
    cookie_session["project_id"] = project_id

    # 2) Bridge to SSE SessionStore (if available)
    if session_id:
        try:
            s = await SESSIONS.ensure(session_id)
            if s:
                s.project_path = str(project_dir)
                s.meta.setdefault("project", {}).update({"path": str(project_dir), "project_id": project_id})
                ev_project_selected(root=str(project_dir), session_id=session_id, project_id=project_id)
        except Exception:
            # non-fatal; UI will still work with cookie session
            pass

    # Remove transient session if present
    if provided_token:
        try:
            delete_session_shared(aidev_dir, provided_token)
        except Exception:
            pass

    return {
        "ok": True,
        "project": meta,
        "created_files": written_entries,
        "files_written": files_written,
    }


@router.post("/select")
async def select_project(
    req: Request,
    body: SelectProjectBody,
    session_id: Optional[str] = Query(default=None, description="Bridge to SSE SessionStore"),
) -> Dict[str, Any]:
    """
    Select an existing project. Writes to both the cookie session and (if provided) the SSE SessionStore.
    Returns hydrated metadata including app_descrip + compiled markdown, matching the legacy /projects/select shape.
    """
    payload = _validate_payload(body.model_dump(), "project_select.schema.json")
    cookie_session = get_session_store(req)

    project_path = Path(payload["project_path"]).resolve()
    if not project_path.exists() or not project_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_path}")

    # Generate stable project_id (or read from file if present)
    meta_file = project_path / ".aidev.project.json"
    project_id: Optional[str] = None
    if meta_file.exists():
        try:
            project_id = json.loads(meta_file.read_text(encoding="utf-8")).get("project_id")
        except Exception:
            project_id = None
    project_id = project_id or _stable_project_id(project_path)

    # 1) Cookie session
    cookie_session["project_path"] = str(project_path)
    cookie_session["project_id"] = project_id

    # 2) Bridge to SSE SessionStore (if available)
    if session_id:
        try:
            s = await SESSIONS.ensure(session_id)
            if s:
                s.project_path = str(project_path)
                s.meta.setdefault("project", {}).update({"path": str(project_path), "project_id": project_id})
                ev_project_selected(root=str(project_path), session_id=session_id, project_id=project_id)
        except Exception:
            pass

    # Hydrate descriptions from .aidev/app_descrip.txt + project_description.md (with root fallbacks)
    app_description, compiled_md = _read_project_descriptions(project_path)

    return {
        "ok": True,
        "root": str(project_path),
        "project_path": str(project_path),
        "project_id": project_id,
        "project": {
            "path": str(project_path),
            "project_id": project_id,
            "app_description": app_description,
            "project_description_md": compiled_md,
        },
    }


@router.post("/update-descriptions")
async def update_descriptions(
    req: Request,
    body: UpdateDescriptionsBody,
    session_id: Optional[str] = Query(default=None, description="(unused, kept for symmetry)"),
) -> Dict[str, Any]:
    """
    Update the human-authored project description.

    New behavior:
    - Treat .aidev/app_descrip.txt as the user-editable source of truth (with root/app_descrip.txt as legacy fallback).
    - Write ONLY to .aidev/app_descrip.txt.
    - Optionally trigger a brief compile so project_description.md + project_metadata.json
      are refreshed by the central brief pipeline.
    """
    payload = _validate_payload(body.model_dump(), "project_update_descriptions.schema.json")
    cookie_session = get_session_store(req)

    project_path = Path(cookie_session.get("project_path", "")).resolve()
    if not project_path or not project_path.exists():
        raise HTTPException(status_code=400, detail="No active project; select or create one first")

    aidev_dir = project_path / ".aidev"
    _ensure_dir(aidev_dir)
    app_path = aidev_dir / "app_descrip.txt"

    # Read current contents (empty string if missing). Also support legacy root/app_descrip.txt.
    if app_path.exists():
        cur_app = app_path.read_text(encoding="utf-8")
    else:
        legacy_path = project_path / "app_descrip.txt"
        if legacy_path.exists():
            cur_app = legacy_path.read_text(encoding="utf-8")
        else:
            cur_app = ""

    new_app = (payload.get("app_description") or "").strip()
    if not new_app:
        raise HTTPException(status_code=400, detail="app_description cannot be empty")

    # Compute diff for the client (before writing)
    diff_app = _unified_diff(cur_app, new_app, "app_descrip.txt", "app_descrip.txt (updated)")

    # Write the new human-authored description to .aidev/app_descrip.txt
    app_path.write_text(new_app, encoding="utf-8")

    # Optional: immediately recompile the structured brief so UI can use it.
    compiled_info = None
    try:
        brief_text, brief_path = get_or_build(
            project_root=project_path,
            create_if_missing=True,
            force_refresh=True,
            ttl_hours=None,
        )
        compiled_info = {
            "path": brief_path,
            "hash": compute_brief_hash(brief_text),
            "bytes": len(brief_text.encode("utf-8")),
        }
    except Exception as e:
        # Non-fatal: user description is updated even if brief compilation fails.
        compiled_info = {"error": str(e)}

    return {
        "ok": True,
        "app_descrip": {
            "old_len": len(cur_app),
            "new_len": len(new_app),
            "diff": diff_app,
        },
        "compiled_brief": compiled_info,
        "note": (
            ".aidev/app_descrip.txt has been updated. "
            "project_description.md + project_metadata.json are maintained by the brief compiler."
        ),
    }


@router.post("/run-checks")
async def run_checks(
    req: Request,
    body: RunChecksBody,
    session_id: Optional[str] = Query(default=None, description="(unused, kept for symmetry)"),
) -> Dict[str, Any]:
    """
    Detect runtimes and run their checks. Returns per-runtime logs and pass/fail.
    """
    _ = _validate_payload(body.model_dump(), "project_run_checks.schema.json")
    cookie_session = get_session_store(req)

    project_path = Path(body.project_path or cookie_session.get("project_path", "")).resolve()
    if not project_path or not project_path.exists():
        raise HTTPException(status_code=400, detail="No active project; select or create one first")

    if detect_runtimes is None:
        raise HTTPException(status_code=501, detail="Runtime detection not available")

    try:
        runtimes = detect_runtimes(str(project_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_runtimes failed: {e}")

    if not runtimes:
        return {"ok": True, "results": [], "message": "No runtimes detected"}

    results = []
    for rt in runtimes:
        name = getattr(rt, "name", rt.__class__.__name__)
        fn = getattr(rt, "run_checks", None)
        if not callable(fn):
            results.append({"runtime": name, "ok": False, "error": "run_checks() not implemented"})
            continue
        try:
            out = fn(project_path)
            ok = bool(out.get("ok"))
            results.append({"runtime": name, **out, "ok": ok})
        except Exception as e:
            results.append({"runtime": name, "ok": False, "error": str(e)})

    overall_ok = all(r.get("ok") for r in results if "ok" in r)
    return {"ok": overall_ok, "results": results}
