# aidev/routes/targets.py
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..llm_client import LLMClient, select_targets_for_rec
from ..session_store import get_session_store
from ..repo_map import PROJECT_MAP_REL, build_project_map
from runtimes.path_safety import resolve_safe_path
from ..state import TraceLogger

router = APIRouter(prefix="/api", tags=["targets"])


class SelectTargetsRequest(BaseModel):
    """
    Request body for selecting target files for a natural-language recommendation.

    Fields:
      - session_id: optional existing session; a new one will be created if missing
      - recommendation_text: free-form description of the change the user wants
      - project_root: optional explicit project root; defaults to the session's project_path
      - max_tokens: soft cap for model completion tokens
      - model: optional override for the model name
    """
    session_id: Optional[str] = None
    recommendation_text: str
    project_root: Optional[str] = None
    max_tokens: Optional[int] = 600
    model: Optional[str] = None


class SelectTargetsResponse(BaseModel):
    ok: bool
    session_id: str
    project_root: str
    paths: List[str]
    usage: dict


@router.post("/select-targets", response_model=SelectTargetsResponse)
async def api_select_targets(req: SelectTargetsRequest) -> SelectTargetsResponse:
    """
    Convenience API to ask the LLM which files in a project should be targeted
    for a given natural-language recommendation.

    This delegates to LLMClient.select_targets_for_rec(), which is responsible
    for:
      - loading the system.target_select.md prompt
      - applying the aidev/schemas/targets.schema.json schema
      - extracting a List[str] of target paths from the richer planner output
    """
    sessions = get_session_store()
    session = await sessions.ensure(req.session_id)

    # Persist project path on the session for subsequent calls, if provided.
    if req.project_root:
        session.project_path = req.project_root

    project_root = req.project_root or session.project_path
    if not project_root:
        # Fall back to CWD if nothing known; keeps API usable.
        project_root = str(Path.cwd())

    try:
        client = LLMClient(model=req.model)
        targets, resp = select_targets_for_rec(
            client=client,
            recommendation_text=req.recommendation_text,
            project_root=project_root,
            max_tokens=req.max_tokens or 600,
        )

        paths: List[str] = []
        if isinstance(targets, list):
            for t in targets:
                if isinstance(t, dict):
                    p = t.get("path")
                    if isinstance(p, str) and p.strip():
                        paths.append(p.strip())

        # de-dupe while preserving order
        seen = set()
        paths = [p for p in paths if not (p in seen or seen.add(p))]

        return SelectTargetsResponse(
            ok=True,
            session_id=session.id,
            project_root=project_root,
            paths=paths,
            usage={
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
                "model": resp.model or client.model,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"select-targets failed: {e}")


# ---- Collect-files flow: list, preview, approve --------------------------------

# Reasonable default bundle cap (bytes). Clients will get a helpful error
# if the selection exceeds this; server-side bundling should enforce limits.
BUNDLE_SIZE_LIMIT = 5 * 1024 * 1024  # 5 MB


class CollectFilesListRequest(BaseModel):
    project_root: Optional[str] = None
    # optional hint: prefer small files / first_n defaults
    first_n: Optional[int] = 50


class FileEntry(BaseModel):
    path: str
    size: int
    sha256: str
    last_modified: str
    # When returning an approved bundle include the exact file content
    content: Optional[str] = None


class CollectFilesListResponse(BaseModel):
    ok: bool
    project_root: str
    files: List[FileEntry]
    default_selection: List[str]


def _sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_entry_from_path(p: Path, project_root: Path) -> Optional[Dict]:
    try:
        stat = p.stat()
    except Exception:
        return None
    rel = str(p.relative_to(project_root))
    return {
        "path": rel,
        "size": stat.st_size,
        "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256_of_path(p),
    }


def _read_project_map_files(project_root: Path) -> Optional[List[str]]:
    """
    Load the canonical v2 project_map and return its file paths.

    Preferred:
      - Use repo_map.build_project_map(project_root) to build/refresh the
        LLM-facing .aidev/project_map.json from cards.

    Fallback:
      - If build_project_map fails for any reason, try to read the existing
        .aidev/project_map.json directly and extract files[*].path.

    Returns:
      - List[str] of repo-relative file paths in the map, or None if unavailable.
    """
    root = project_root.resolve()

    data = None

    # Preferred: use the v2 builder (idempotent; only rewrites on change).
    try:
        data = build_project_map(root, force=False)
    except Exception:
        data = None

    # Fallback: read whatever is already on disk if builder failed.
    if not isinstance(data, dict):
        pm = root / PROJECT_MAP_REL
        if not pm.exists():
            return None
        try:
            with pm.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

    files: List[str] = []
    for fi in data.get("files") or []:
        if not isinstance(fi, dict):
            continue
        p = fi.get("path")
        if isinstance(p, str) and p:
            files.append(p)

    return files or None


@router.post("/workspaces/collect-files", response_model=CollectFilesListResponse)
async def api_collect_files_list(req: CollectFilesListRequest) -> CollectFilesListResponse:
    """
    Return a canonical list of files under a project root suitable for
    user selection. When .aidev/project_map.json is present prefer its ordering;
    otherwise do a lightweight discovery.

    Each file has path (relative to root), size, last_modified (ISO), and sha256.
    A sensible default_selection (list of paths) is included to help the UI.
    """
    project_root = req.project_root or str(Path.cwd())
    root = Path(project_root).resolve()

    # Try project map first (preferred canonical source)
    map_files = _read_project_map_files(root)
    files: List[Dict] = []

    if map_files:
        for rel in map_files:
            p = (root / rel).resolve()
            # only include files that actually exist and are under root
            try:
                p.relative_to(root)
            except Exception:
                continue
            entry = _file_entry_from_path(p, root)
            if entry:
                files.append(entry)
    else:
        # Fallback discovery: walk root and collect regular files, skipping .aidev and .git
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Skip internal .aidev and VCS dirs
            if ".aidev" in p.parts or ".git" in p.parts:
                continue
            entry = _file_entry_from_path(p, root)
            if entry:
                files.append(entry)

    # Sort files: prefer smaller files first to make selection quicker for users
    files.sort(key=lambda x: (x["size"], x["path"]))

    first_n = req.first_n or 50
    default_selection = [f["path"] for f in files[:min(first_n, len(files))]]

    return CollectFilesListResponse(
        ok=True,
        project_root=str(root),
        files=[FileEntry(**f) for f in files],
        default_selection=default_selection,
    )


class PreviewRequest(BaseModel):
    project_root: Optional[str] = None
    path: str


class PreviewResponse(BaseModel):
    ok: bool
    project_root: str
    path: str
    content: str


@router.post("/workspaces/collect-files/preview", response_model=PreviewResponse)
async def api_collect_files_preview(req: PreviewRequest) -> PreviewResponse:
    """
    Return the full, exact file content for a single path. Path containment
    is enforced: the resolved path must be within the provided project_root.
    """
    project_root = req.project_root or str(Path.cwd())
    root = Path(project_root).resolve()

    try:
        abs_path = resolve_safe_path(req.path, root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    p = Path(abs_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    try:
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
    except Exception:
        # Fallback to binary -> latin-1 decode for faithful round-trip
        text = p.read_bytes().decode("latin-1")

    return PreviewResponse(ok=True, project_root=str(root), path=str(Path(req.path)), content=text)


class ApproveRequest(BaseModel):
    project_root: Optional[str] = None
    selected_paths: List[str]
    note: Optional[str] = None


class ApproveResponse(BaseModel):
    ok: bool
    project_root: str
    manifest: List[FileEntry]
    total_size: int


def _atomic_append_trace(project_root: Path, event: Dict) -> None:
    trace_file = project_root / ".aidev" / "trace.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    # use append with fsync for durability
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def build_analysis_bundle(project_root: Path, selected_paths: List[str], max_bytes: int) -> Dict:
    """
    Build a server-side bundle for the given selected paths.

    Returns a dict with either:
      { "bundle": { "files": [...], "total_size": int } }
    or
      { "oversize": True, "total_size": int, "max_bytes": int, "suggestions": {...} }

    Each file entry in bundle.files contains: path (repo-relative), content (string,
    decoded the same way preview uses), size (bytes), sha256 (hex of raw bytes).
    """
    root = project_root.resolve()
    entries: List[Dict] = []
    total = 0

    for rel in selected_paths:
        try:
            abs_path = resolve_safe_path(rel, root)
        except ValueError as e:
            raise ValueError(f"invalid path: {rel}: {e}")
        p = Path(abs_path)
        if not p.exists() or not p.is_file():
            raise ValueError(f"file not found: {rel}")

        # Read raw bytes for exact size and sha256
        try:
            raw = p.read_bytes()
        except Exception as e:
            raise ValueError(f"failed to read file {rel}: {e}")

        h = hashlib.sha256()
        h.update(raw)
        sha = h.hexdigest()
        size = len(raw)

        # Decode content the same way preview does so clients can compare
        try:
            content = raw.decode("utf-8")
        except Exception:
            # match preview fallback
            content = raw.decode("latin-1")

        entries.append({
            "path": str(Path(rel)),
            "content": content,
            "size": size,
            "sha256": sha,
        })
        total += size

    if total > max_bytes:
        # Prepare suggestions to help narrow selection: largest files and counts
        largest = sorted(entries, key=lambda e: e["size"], reverse=True)[:10]
        suggestions = {
            "total_selected_files": len(entries),
            "total_size": total,
            "max_bytes": max_bytes,
            "top_largest": [{"path": e["path"], "size": e["size"]} for e in largest],
        }
        return {"oversize": True, "total_size": total, "max_bytes": max_bytes, "suggestions": suggestions}

    return {"bundle": {"files": entries, "total_size": total}}


@router.post("/workspaces/collect-files/approve", response_model=ApproveResponse)
async def api_collect_files_approve(req: ApproveRequest) -> ApproveResponse:
    """
    Accept the final selection, construct a server-side bundle manifest (no
    LLM calls), record an approval trace entry to .aidev/trace.jsonl, and
    return bundle metadata. If total selected size exceeds a configured limit
    return an error instead of constructing the bundle.
    """
    project_root = req.project_root or str(Path.cwd())
    root = Path(project_root).resolve()

    try:
        res = build_analysis_bundle(root, req.selected_paths, BUNDLE_SIZE_LIMIT)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if res.get("oversize"):
        sugg = res.get("suggestions", {})
        # Do not write any trace entry when refusing to send due to oversize
        raise HTTPException(status_code=400, detail={
            "error": "selected files exceed size limit",
            "total_size": res.get("total_size"),
            "max_bytes": res.get("max_bytes"),
            "suggestions": sugg,
        })

    bundle = res.get("bundle")
    files = bundle.get("files", [])
    total = bundle.get("total_size", 0)

    # Record approval trace entry using TraceLogger
    try:
        tl = TraceLogger(root)
        tl.write(
            kind="analysis_bundle",
            action="analysis_bundle_send",
            payload={
                "selected_paths": [f["path"] for f in files],
                "total_size": total,
                "outcome": "sent",
                "note": req.note,
            },
        )
    except Exception:
        # Best-effort: do not fail the approval if tracing can't be written,
        # but surface an HTTP 500 if atomic trace is a hard requirement would be different.
        pass

    manifest = [
        {
            "path": f["path"],
            "size": f["size"],
            "sha256": f["sha256"],
            "last_modified": "",
            "content": f.get("content"),
        }
        for f in files
    ]

    return ApproveResponse(
        ok=True,
        project_root=str(root),
        manifest=[FileEntry(**m) for m in manifest],
        total_size=total,
    )
