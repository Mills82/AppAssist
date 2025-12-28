# aidev/routes/workspaces.py
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Header, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import logger
from ..events import (
    status as ev_status,
    done as ev_done,
    project_selected as ev_project_selected,
)

# Session + project discovery
from ..session_store import get_session_store, SessionStore
from ..workspace import find_projects, ProjectCandidate
from ..discovery import scan_workspace_projects

# Structure/cards
from ..structure import discover_structure, compact_structure
from ..cards import KnowledgeBase
from ..config import load_project_config

router = APIRouter()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def _stable_project_id(p: Path) -> str:
    """
    Stable UUID (v5) derived from the absolute, lowercased POSIX path.
    """
    key = p.expanduser().resolve().as_posix().lower()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"file://{key}"))


class ProjectCandidateModel(BaseModel):
    path: str
    score: int = 0
    kind: Optional[str] = None
    markers: List[str] = Field(default_factory=list)
    # Stable id + grouped children count (shown in UI)
    project_id: Optional[str] = None
    children_count: int = 0

    @classmethod
    def from_cand(cls, c: ProjectCandidate) -> "ProjectCandidateModel":
        pid = _stable_project_id(Path(c.path))
        return cls(
            path=str(c.path),
            score=c.score,
            kind=c.kind,
            markers=c.markers,
            project_id=pid,
            children_count=getattr(c, "children_count", 0) or 0,
        )


class SelectRequest(BaseModel):
    session_id: Optional[str] = None
    path: Optional[str] = None
    project_path: Optional[str] = None


class SelectResponse(BaseModel):
    session_id: str
    selected: ProjectCandidateModel
    project_id: Optional[str] = None


class StructureResponse(BaseModel):
    path: str
    bytes: int
    structure: Optional[Dict[str, Any]] = None  # omitted if return_map=false


class CardsIndexResponse(BaseModel):
    cards_index: Dict[str, Any]


class RefreshCardsRequest(BaseModel):
    session_id: str
    force: bool = False
    # Optional per-request override of the project root; does NOT persist to session
    project_path: Optional[str] = None


class SummarizeChangedRequest(BaseModel):
    session_id: str
    changed_only: bool = True
    model: Optional[str] = None


class SummarizeChangedResponse(CardsIndexResponse):
    summarized_count: int
    skipped_count: int


class EnrichRequest(BaseModel):
    session_id: str
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus text to guide enrichment.",
    )
    enrich_top_k: int = Field(
        default=24,
        ge=0,
        le=200,
        description="Cross-file selection budget.",
    )
    ttl_days: int = Field(
        default=365,
        ge=0,
        le=365,
        description="Skip re-enriching fresh cards.",
    )
    no_protect_llm: bool = Field(
        default=False,
        description="If true, disable token-protection measures.",
    )


class EnrichResponse(CardsIndexResponse):
    enriched: bool = True
    summarized_count: int = 0
    skipped_count: int = 0


# Orchestrated AI cards request/response
class AICardsRequest(BaseModel):
    session_id: str
    mode: Literal["changed", "deep", "full"] = "changed"
    model: Optional[str] = None
    ttl_days: int = Field(default=365, ge=0, le=365)
    protect_llm: bool = True
    enrich: Optional[bool] = None  # if omitted, inferred from mode (deep/full => True)
    enrich_top_k: int = Field(default=24, ge=0, le=200)
    focus: Optional[str] = None
    # Optional per-request override of the project root; does NOT persist to session
    project_path: Optional[str] = None


class AICardsResponse(CardsIndexResponse):
    summarized_count: int = 0
    skipped_count: int = 0
    enriched: bool = False


# Optional deep notes response (if supported by KnowledgeBase)
class DeepNotesResponse(CardsIndexResponse):
    notes: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helpers (index & cards reading, normalized across versions)
# -----------------------------------------------------------------------------

PROJECT_MARKERS = [".git", "pyproject.toml", "package.json", ".aidev", "app_descrip.txt"]


def _prefer_repos_ancestor(p: Path) -> Path:
    """If 'p' is inside a folder named 'Repos' (case-insensitive), return that ancestor; else return p."""
    for anc in [p] + list(p.parents):
        if anc.name.lower() == "repos":
            return anc
    return p


def _guess_default_workspace_root() -> Path:
    """
    Default to the user's Repos folder if we can find it.
    Order:
      1) AIDEV_WORKSPACE_ROOT env
      2) If CWD is under a 'Repos' folder, return that 'Repos'
      3) ~/Repos or ~/repos if it exists
      4) CWD as a last resort
    """
    env = os.getenv("AIDEV_WORKSPACE_ROOT")
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            pass

    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if parent.name.lower() == "repos":
            return parent

    home = Path.home()
    for candidate in (home / "Repos", home / "repos"):
        if candidate.exists():
            return candidate.resolve()

    return cwd


def _project_root_or_400(session: Any) -> Path:
    project_root = (
        Path(getattr(session, "project_path", "")).resolve()
        if getattr(session, "project_path", None)
        else None
    )
    if not project_root:
        raise HTTPException(status_code=400, detail="No project selected for this session.")
    if not project_root.exists():
        raise HTTPException(status_code=400, detail=f"Project root not found: {project_root}")
    return project_root


def _resolve_request_project_root(
    session: Any,
    project_path: Optional[str],
    header_project: Optional[str],
    source_label: str = "project_path",
) -> Tuple[Path, str]:
    """
    Resolve a project root for a request with preference:
      1) explicit project_path (query/body)
      2) X-AIDEV-PROJECT header
      3) session-selected project

    Returns (Path, source) where source is one of 'query', 'body', 'header', 'session'.
    Raises HTTPException(400) if the resolved path does not exist or is not a dir.
    """
    if project_path:
        try:
            p = Path(project_path).expanduser().resolve()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid project path: {project_path}")
        src = "body" if source_label == "body" else "query"
    elif header_project:
        try:
            p = Path(header_project).expanduser().resolve()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid X-AIDEV-PROJECT header: {header_project}")
        src = "header"
    else:
        p = _project_root_or_400(session)
        src = "session"

    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Project root not found: {p}")

    # Small debug/log line to show which source provided the project root
    try:
        logger.info("resolved_project_root", ctx={"source": src, "project_root": str(p)})
    except Exception:
        # best-effort; don't fail request on logging
        pass

    return p, src


# ---- index normalization ----

def _normalize_cards_index(idx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize any historical/variant cards index shape into:
      { "nodes": [<rel paths>], "cards": { "<rel>": { ...entry... } } }

    This uses KnowledgeBase._ensure_summary_obj so summary/AI/staleness
    semantics match cards.py exactly.
    """
    from ..cards import KnowledgeBase as _KB  # local import to avoid cycles in some loaders

    if not isinstance(idx, dict) or not idx:
        return {"nodes": [], "cards": {}}

    def _ensure(meta: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize the summary in place so callers get full meta, not just summary.
        _KB._ensure_summary_obj(meta)  # type: ignore[attr-defined]
        return meta

    # Canonical v2: {"cards": {...}, ...} with optional "nodes"
    if "cards" in idx and isinstance(idx["cards"], dict):
        cards: Dict[str, Dict[str, Any]] = {}
        for rel, meta in idx["cards"].items():
            rel_posix = Path(str(rel)).as_posix()
            cards[rel_posix] = _ensure(dict(meta) if isinstance(meta, dict) else {})

        raw_nodes = idx.get("nodes")
        if isinstance(raw_nodes, list):
            nodes = [Path(str(x)).as_posix() for x in raw_nodes if isinstance(x, str)]
        else:
            nodes = sorted(cards.keys())

        return {"nodes": nodes, "cards": cards}

    # Already normalized legacy form: { "nodes": [...], "cards": {...} }
    if (
        "nodes" in idx
        and "cards" in idx
        and isinstance(idx["nodes"], list)
        and isinstance(idx["cards"], dict)
    ):
        nodes = [Path(str(x)).as_posix() for x in idx["nodes"] if isinstance(x, str)]
        cards: Dict[str, Dict[str, Any]] = {}
        for k, v in idx["cards"].items():
            rel = Path(k).as_posix()
            meta = dict(v) if isinstance(v, dict) else {}
            cards[rel] = _ensure(meta)
        return {"nodes": nodes, "cards": cards}

    # project_map.json-like: { "files": [ { "path": "..." }, ... ] }
    if "files" in idx and isinstance(idx["files"], list):
        nodes: List[str] = []
        cards: Dict[str, Dict[str, Any]] = {}
        for f in idx["files"]:
            if not isinstance(f, dict):
                continue
            rel = f.get("path")
            if not rel:
                continue
            rel = Path(str(rel)).as_posix()
            meta: Dict[str, Any] = {
                "path": rel,
                "kind": f.get("kind") or "",
                "language": f.get("language") or "other",
                # carry through v2 hints if present
                "tags": f.get("tags") or [],
                "size": int(f.get("size") or 0),
                "sha256": f.get("sha256") or "",
                "chunks": f.get("chunks") or [],
                "summary": f.get("summary") or f.get("prompt") or "",
                "imports": [],
                "imports_resolved": [],
                "exports": [],
                "symbols": [],
                "routes": f.get("routes") or [],
                "cli_args": f.get("cli_args") or [],
                "env_vars": f.get("env_vars") or [],
                "owners": [],
                "git_last_ts": None,
                "git_last_author": None,
                "embedding": None,
                "file_sha": "",
                "sha_changed": False,
                # respect project_map.changed if present
                "changed": bool(f.get("changed") or False),
                "needs_ai_refresh": False,
            }
            cards[rel] = _ensure(meta)
            nodes.append(rel)
        return {"nodes": sorted(nodes), "cards": cards}

    # Flat dict: keys are rel paths
    if all(isinstance(v, dict) for v in idx.values()):
        nodes: List[str] = []
        cards: Dict[str, Dict[str, Any]] = {}
        for k, v in idx.items():
            rel = Path(k).as_posix()
            meta = dict(v) if isinstance(v, dict) else {}
            cards[rel] = _ensure(meta)
            nodes.append(rel)
        return {"nodes": sorted(nodes), "cards": cards}

    # Unknown -> empty normalized
    return {"nodes": [], "cards": {}}


def _read_cards_index(project_root: Path) -> Dict[str, Any]:
    """
    Read/construct the cards index and normalize it.

    Preferred (canonical):
      - .aidev/cards/index.json  (owned by KnowledgeBase.save_card_index)

    Fallbacks (in order):
      - Reconstruct from per-file .aidev/cards/**/*.card.json
      - Legacy: .aidev/cards.json, .cards.json
      - Last resort: synthesize from .aidev/project_map.json (files[])

    NOTE: .aidev/index.json is a legacy *graph* snapshot and is no longer
    considered a cards index source.
    """

    def _minimal_normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept multiple shapes and return {"nodes": [...], "cards": {...}} with
        normalized summaries. This is a safety net; when present, the global
        _normalize_cards_index will be preferred so semantics match cards.py.
        """
        from ..cards import KnowledgeBase as _KB  # local import

        def _ensure(meta: Dict[str, Any]) -> Dict[str, Any]:
            _KB._ensure_summary_obj(meta)  # type: ignore[attr-defined]
            return meta

        cards: Dict[str, Dict[str, Any]] = {}

        if isinstance(payload, dict) and "cards" in payload and isinstance(payload["cards"], dict):
            for rel, meta in payload["cards"].items():
                rel_posix = Path(str(rel)).as_posix()
                cards[rel_posix] = _ensure(dict(meta) if isinstance(meta, dict) else {})
        elif isinstance(payload, dict) and "files" in payload and isinstance(payload["files"], list):
            # project_map.json → synthesize minimal card metas
            for f in payload["files"]:
                if not isinstance(f, dict):
                    continue
                rel = (f or {}).get("path") or (f or {}).get("rel") or (f or {}).get("name")
                if not rel:
                    continue
                rel_posix = Path(str(rel)).as_posix()
                meta: Dict[str, Any] = {
                    "path": rel_posix,
                    "kind": (f or {}).get("kind") or "",
                    "language": (f or {}).get("language") or "other",
                    "tags": (f or {}).get("tags") or [],
                    "size": int((f or {}).get("size") or 0),
                    "sha256": (f or {}).get("sha256") or "",
                    "chunks": (f or {}).get("chunks") or [],
                    "summary": (f or {}).get("summary") or "",
                    "imports": [],
                    "imports_resolved": [],
                    "exports": [],
                    "symbols": [],
                    "routes": (f or {}).get("routes") or [],
                    "cli_args": (f or {}).get("cli_args") or [],
                    "env_vars": (f or {}).get("env_vars") or [],
                    "owners": [],
                    "git_last_ts": None,
                    "git_last_author": None,
                    "embedding": None,
                    "file_sha": "",
                    "sha_changed": False,
                    # respect project_map.changed if present
                    "changed": bool((f or {}).get("changed") or False),
                    "needs_ai_refresh": False,
                }
                cards[rel_posix] = _ensure(meta)
        elif isinstance(payload, dict):
            # Assume flat mapping: { rel_path: meta, ... }
            for rel, meta in payload.items():
                rel_posix = Path(str(rel)).as_posix()
                cards[rel_posix] = _ensure(dict(meta) if isinstance(meta, dict) else {})
        else:
            cards = {}

        return {"nodes": sorted(cards.keys()), "cards": cards}

    # Use project-local normalizer if present; fallback to our minimal one.
    normalize = globals().get("_normalize_cards_index", _minimal_normalize)  # type: ignore

    # 1) Canonical index
    p = project_root / ".aidev" / "cards" / "index.json"
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return normalize(raw)
        except Exception:
            pass

    # 2) Rebuild from per-file cards (*.card.json)
    cards_dir = project_root / ".aidev" / "cards"
    if cards_dir.exists():
        collected: Dict[str, Dict[str, Any]] = {}
        for fp in cards_dir.rglob("*.card.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                rel = (
                    raw.get("path")
                    or fp.relative_to(cards_dir).as_posix().removesuffix(".card.json")
                )
                if rel:
                    collected[Path(str(rel)).as_posix()] = raw
            except Exception:
                continue
        if collected:
            return normalize({"cards": collected})

    # 3) Legacy card indexes
    for legacy in (
        project_root / ".aidev" / "cards.json",
        project_root / ".cards.json",
    ):
        if legacy.exists():
            try:
                raw = json.loads(legacy.read_text(encoding="utf-8"))
                return normalize(raw)
            except Exception:
                continue

    # 4) Last resort: synthesize from project_map.json
    pm = project_root / ".aidev" / "project_map.json"
    if pm.exists():
        try:
            raw = json.loads(pm.read_text(encoding="utf-8"))
            return _minimal_normalize(raw)
        except Exception:
            pass

    return {"nodes": [], "cards": {}}


def _get_index_entry(idx: Dict[str, Any], rel_path: str) -> Dict[str, Any]:
    """
    Works on normalized shape only.
    """
    if not isinstance(idx, dict):
        return {}
    posix = Path(rel_path).as_posix()
    cards = idx.get("cards") or {}
    if not isinstance(cards, dict):
        return {}
    return cards.get(posix, {}) or {}


def _iter_index_nodes(idx: Dict[str, Any]) -> Iterable[str]:
    """
    Iterate normalized nodes.
    """
    if not isinstance(idx, dict):
        return []
    nodes = idx.get("nodes")
    if isinstance(nodes, list):
        for rel in nodes:
            if isinstance(rel, str):
                yield rel
    return []


# ---- per-file card reading (normalized view) ----

def _pick_summary_text(card: Dict[str, Any]) -> str:
    """
    Readers should prefer AI text and gracefully fall back to heuristic:
      - preferred: card["summary"]["ai_text"]
      - fallback:  card["summary"]["heuristic"]
    """
    if not isinstance(card, dict):
        return ""
    s = card.get("summary")
    if isinstance(s, dict):
        ai = s.get("ai_text")
        if isinstance(ai, str) and ai.strip():
            return ai.strip()
        he = s.get("heuristic")
        if isinstance(he, str) and he.strip():
            return he.strip()
    return ""


def _normalize_card(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single card JSON to a stable shape and attach 'summary_text' for convenience.
    Also flatten staleness.changed to a top-level 'changed' flag for consumers.
    """
    if not isinstance(card, dict):
        return {"summary_text": ""}

    out = dict(card)
    out["path"] = Path(str(out.get("path") or out.get("rel") or "")).as_posix()
    out["summary_text"] = _pick_summary_text(out)

    # Expose a flat changed flag derived from staleness.changed
    st = out.get("staleness")
    if isinstance(st, dict):
        out["changed"] = bool(st.get("changed", False))
    else:
        out["changed"] = False

    return out


def _read_single_card(project_root: Path, rel_path: str) -> Dict[str, Any]:
    """
    Read one .card.json and return a normalized shape with 'summary_text' selected.
    """
    card_path = (
        project_root
        / ".aidev"
        / "cards"
        / f"{Path(rel_path).as_posix()}.card.json"
    )
    if not card_path.exists():
        return {}
    try:
        raw = json.loads(card_path.read_text(encoding="utf-8"))
        return _normalize_card(raw)
    except Exception:
        return {}


# ---- AI summary extraction for orchestration counters ----

def _extract_ai_fields(entry: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (ai_sha, ai_text) for *AI* summaries.
    IMPORTANT: Heuristic-only summaries do NOT count as AI here.
      - Uses entry["summary"]["ai_text"] as the AI text
      - Uses entry["summary"]["ai_sha"] (if present) as the hash
    """
    if not isinstance(entry, dict):
        return ("", "")

    s = entry.get("summary")
    if not isinstance(s, dict):
        return ("", "")

    ai_text = s.get("ai_text")
    if isinstance(ai_text, str) and ai_text.strip():
        raw_sha = s.get("ai_sha")
        sha = "" if raw_sha is None else str(raw_sha)
        return (sha, ai_text.strip())

    # Heuristic-only summaries should not count as AI
    return ("", "")


def _snapshot_ai_index(idx: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    snap: Dict[str, Tuple[str, str]] = {}
    if not isinstance(idx, dict):
        return snap
    for rel in _iter_index_nodes(idx):
        entry = _get_index_entry(idx, rel)
        if not isinstance(entry, dict):
            continue
        ai_sha, ai_text = _extract_ai_fields(entry)
        snap[rel] = (ai_sha, ai_text)
    return snap


def _diff_summary_counts_index(
    after_idx: Dict[str, Any],
    before_snap: Dict[str, Tuple[str, str]],
) -> Tuple[int, int]:
    nodes = list(_iter_index_nodes(after_idx))
    summarized = 0
    for rel in nodes:
        entry = _get_index_entry(after_idx, rel)
        if not isinstance(entry, dict):
            continue
        ai_sha, ai_text = _extract_ai_fields(entry)
        if not ai_text:
            continue
        prev = before_snap.get(rel)
        if prev is None:
            summarized += 1
        else:
            prev_sha, prev_text = prev
            if ai_sha != prev_sha or ai_text != prev_text:
                summarized += 1
    skipped = max(0, len(nodes) - summarized)
    return summarized, skipped


def _has_cards_baseline(project_root: Path) -> bool:
    idx = _read_cards_index(project_root)
    if any(True for _ in _iter_index_nodes(idx)):
        return True
    # Per-file cards exist?
    cards_dir = project_root / ".aidev" / "cards"
    try:
        for _p in cards_dir.rglob("*.card.json"):
            return True
    except Exception:
        pass
    return False


# -----------------------------------------------------------------------------
# Error helpers
# -----------------------------------------------------------------------------

def _trace_tail(exc: BaseException | None = None, n: int = 25) -> str:
    try:
        if exc is None:
            lines = traceback.format_exc().splitlines()
            return "\n".join(lines[-n:])
        lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return "".join(lines[-n:])
    except Exception:
        return ""


def _json_error(
    *,
    project_root: Optional[Path],
    where: str,
    status: int,
    message: str,
    exc: BaseException | None = None,
) -> JSONResponse:
    idx = _read_cards_index(project_root) if project_root else {"nodes": [], "cards": {}}
    payload = {
        "error": message,
        "where": where,
        "trace": _trace_tail(exc),
        "cards_index": idx,
    }
    return JSONResponse(status_code=status, content=payload)


# ---- Progress helpers --------------------------------------------------------

def _emit_ai_progress(
    session_id: str,
    stage: str,
    i: int,
    n: int,
    file_path: str = "",
    detail: str = "",
) -> None:
    pct = 0 if n <= 0 else int((i / n) * 100)
    ev_status(
        stage,
        session_id=session_id,
        stage=stage,
        progress_pct=pct,
        i=i,
        n=n,
        file=file_path,
        detail=detail,
    )


def _progress_cb_factory(session_id: str, stage: str):
    """
    Returns a tolerant callback that tries to extract (i, n, file, detail) from either
    positional args or kwargs. Safe to pass even if the callee ignores it.
    """

    def _cb(*args, **kwargs):
        try:
            i = kwargs.get("i")
            n = kwargs.get("n")
            file_path = kwargs.get("file") or kwargs.get("path") or ""
            detail = kwargs.get("detail") or ""
            # Try positional fallbacks
            if i is None and len(args) >= 1 and isinstance(args[0], int):
                i = args[0]
            if n is None and len(args) >= 2 and isinstance(args[1], int):
                n = args[1]
            if not file_path and len(args) >= 3 and isinstance(args[2], str):
                file_path = args[2]
            if i is not None and n is not None:
                _emit_ai_progress(
                    session_id,
                    stage,
                    int(i),
                    int(n),
                    str(file_path),
                    str(detail),
                )
        except Exception:
            # progress is best-effort; never raise
            pass

    return _cb


_ENRICH_PROG_RE = re.compile(r"^\s*PROGRESS\s+(\d+)\s*/\s*(\d+)\s+(.*)$")


def _try_emit_progress_from_line(
    session_id: str,
    line: str,
    stage: str = "AI deep",
) -> bool:
    """
    Parse a line from the enrich subprocess. Supports either:
      - JSON object with keys i, n, file (optionally 'detail')
      - 'PROGRESS i/n path'
    Returns True if a progress event was emitted.
    """
    s = (line or "").strip()
    if not s:
        return False
    # JSON?
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            i = int(obj.get("i"))
            n = int(obj.get("n"))
            f = str(obj.get("file") or obj.get("path") or "")
            detail = str(obj.get("detail") or "")
            if i and n:
                _emit_ai_progress(
                    session_id,
                    stage,
                    i,
                    n,
                    f,
                    detail or "enrich (deep)",
                )
                return True
        except Exception:
            pass
    # PROGRESS i/n path
    m = _ENRICH_PROG_RE.match(s)
    if m:
        try:
            i = int(m.group(1))
            n = int(m.group(2))
            f = m.group(3).strip()
            _emit_ai_progress(
                session_id,
                stage,
                i,
                n,
                f,
                "enrich (deep)",
            )
            return True
        except Exception:
            return False
    return False


# -----------------------------------------------------------------------------
# Internal Orchestrator
# -----------------------------------------------------------------------------

def _orchestrate_ai_cards(
    *,
    project_root: Path,
    session_id: str,
    mode: Literal["changed", "deep", "full"],
    model: Optional[str],
    ttl_days: int,
    protect_llm: bool,
    do_enrich: bool,
    enrich_top_k: int,
    focus: Optional[str],
) -> Tuple[Dict[str, Any], int, int, bool]:
    """
    Run summarize (incremental/full as per mode) and optional enrich in one flow.

    All AI work is funneled through KnowledgeBase.generate_ai_summaries +
    cards/index.json.

    Returns: (cards_index, summarized_count, skipped_count, enriched_bool)
    """
    # --- Ensure baseline
    try:
        cfg, _cfg_path = load_project_config(project_root, None)
        includes = list(cfg.get("discovery", {}).get("includes", []))
        excludes = list(cfg.get("discovery", {}).get("excludes", []))
        struct, _ = discover_structure(
            project_root,
            includes,
            excludes,
            max_total_kb=128,
            strip_comments=False,
        )
        kb = KnowledgeBase(project_root, struct)
    except Exception as e:
        raise RuntimeError(f"Failed to build KB: {type(e).__name__}: {e}")

    if not _has_cards_baseline(project_root):
        try:
            kb.update_cards(force=False, changed_only=False)
        except Exception as seed_err:
            logger.warning(
                "baseline build before ai/cards failed (continuing)",
                ctx={"err": type(seed_err).__name__},
                exc=seed_err,
            )

    # --- Summarize pass (incremental by default, full if mode=full)
    before_idx = _read_cards_index(project_root)
    before_snap = _snapshot_ai_index(before_idx)

    # changed_only if we already have *AI* summaries and not full
    has_any_ai = any(
        _extract_ai_fields(_get_index_entry(before_idx, rel))[1]
        for rel in _iter_index_nodes(before_idx)
    )
    changed_only_flag = (mode != "full") and has_any_ai

    ev_status(
        "AI summarize: start",
        session_id=session_id,
        changed_only=changed_only_flag,
        model=(model or ""),
        progress_pct=5,
    )

    total_targets = 0
    try:
        list_targets = getattr(kb, "list_ai_summary_targets", None)
        if callable(list_targets):
            targets = list(list_targets(changed_only=changed_only_flag))
            total_targets = len(targets or [])
        else:
            total_targets = len(list(_iter_index_nodes(before_idx)))
    except Exception:
        total_targets = 0

    if total_targets > 0:
        _emit_ai_progress(
            session_id,
            "AI summarize",
            0,
            total_targets,
            "",
            "start",
        )

    progress_cb = _progress_cb_factory(session_id, "AI summarize")
    try:
        # Prefer progress-capable signature if available
        result = kb.generate_ai_summaries(
            changed_only=changed_only_flag,
            model=model,
            progress_cb=progress_cb,  # type: ignore[call-arg]
        )
    except TypeError:
        # Older KnowledgeBase.generate_ai_summaries without progress_cb
        result = kb.generate_ai_summaries(
            changed_only=changed_only_flag,
            model=model,
        )

    if total_targets > 0:
        _emit_ai_progress(
            session_id,
            "AI summarize",
            total_targets,
            total_targets,
            "",
            "done",
        )

    after_idx = _read_cards_index(project_root)
    summarized_count, skipped_count = _diff_summary_counts_index(
        after_idx,
        before_snap,
    )

    # Allow newer generate_ai_summaries to return richer info, but keep the diff
    if isinstance(result, dict):
        summarized_count = int(result.get("summarized_count", summarized_count))
        skipped_count = int(result.get("skipped_count", skipped_count))
    elif isinstance(result, tuple) and len(result) >= 2:
        try:
            summarized_count = int(result[0])
            skipped_count = int(result[1])
        except Exception:
            pass

    ev_status(
        "AI summarize: done",
        session_id=session_id,
        progress_pct=100,
    )

    # --- Optional enrich pass (cross-file). Respect protect_llm (do not overwrite summaries).
    enriched = False
    if do_enrich:
        ev_status(
            "AI deep: start",
            session_id=session_id,
            focus=(focus or None),
            enrich_top_k=int(enrich_top_k),
            progress_pct=5,
        )
        cmd = [
            sys.executable,
            "-m",
            "aidev_gen_cards",
            "--project-root",
            str(project_root),
            "--enrich",
            "--enrich-top-k",
            str(int(enrich_top_k)),
            "--ttl-days",
            str(int(ttl_days)),
        ]
        if not protect_llm:
            cmd.append("--no-protect-llm")
        if focus:
            cmd.extend(["--focus", focus])

        env = os.environ.copy()
        env["AIDEV_CARD_ENRICH"] = "1"
        try:
            repo_root = Path(__file__).resolve().parents[2]
            prev = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                str(repo_root) if not prev else f"{repo_root}{os.pathsep}{prev}"
            )
        except Exception:
            pass

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        saw_progress = False
        if proc.stdout:
            for line in proc.stdout:
                if _try_emit_progress_from_line(
                    session_id,
                    line,
                    stage="AI deep",
                ):
                    saw_progress = True

        rc = proc.wait()
        if rc != 0:
            tail = f"exit={rc}"
            raise RuntimeError(f"enrich failed (exit {rc}). {tail}")

        # BUGFIX: use session_id, not session.id (no session object in this scope).
        if not saw_progress:
            ev_status(
                "AI deep",
                session_id=session_id,
                progress_pct=100,
            )

        ev_status(
            "AI deep: done",
            session_id=session_id,
            progress_pct=100,
        )
        enriched = True

        after_idx = _read_cards_index(project_root)

    return after_idx, summarized_count, skipped_count, enriched


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@router.get("/workspaces/projects")
def list_projects(
    workspace_root: str | None = Query(default=None),
    group: int = Query(
        default=1,
        ge=0,
        le=1,
        description="1 = group monorepos; 0 = show all projects",
    ),
    depth: int = Query(
        default=8,
        ge=1,
        le=12,
        description="Directory descend depth for project discovery",
    ),
):
    """
    Returns a list of discovered projects.
    """
    try:
        raw_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root
            else _guess_default_workspace_root()
        )
        root = _prefer_repos_ancestor(raw_root)

        if not root.exists():
            raise HTTPException(
                status_code=400,
                detail=f"workspace_root does not exist: {root}",
            )
        if not root.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"workspace_root is not a directory: {root}",
            )

        grouped = bool(group)
        cands = scan_workspace_projects(
            root,
            group=grouped,
            max_depth=int(depth),
        )
        logger.info(
            "scan_projects:done",
            ctx={
                "root": str(root),
                "group": grouped,
                "depth": int(depth),
                "count": len(cands),
            },
        )

        # Normalize + add stable project_id for UI
        for c in cands:
            c.setdefault("kind", "unknown")
            c.setdefault("markers", [])
            c.setdefault("children_count", 0)
            try:
                c["project_id"] = _stable_project_id(Path(c["path"]))
            except Exception:
                c["project_id"] = None

        return {
            "workspace_root": str(root),
            "group": int(grouped),
            "candidates": cands,
        }
    except HTTPException as he:
        return _json_error(
            project_root=None,
            where="projects",
            status=he.status_code,
            message=str(he.detail),
            exc=he,
        )
    except Exception as e:
        return _json_error(
            project_root=None,
            where="projects",
            status=500,
            message=f"scan failed: {type(e).__name__}: {e}",
            exc=e,
        )


@router.post("/workspaces/select", response_model=SelectResponse)
async def select_project(
    body: SelectRequest,
    session_id: Optional[str] = Query(default=None),
    response: Response = None,
    sessions: SessionStore = Depends(get_session_store),
) -> SelectResponse | JSONResponse:
    try:
        # Accept session_id from body OR query
        sid = (body.session_id or session_id or "").strip()

        # Accept path from body.path OR legacy body.project_path
        raw_path = (body.path or body.project_path or "").strip()

        if not sid:
            raise HTTPException(
                status_code=422,
                detail="Missing session_id (provide in body or query).",
            )
        if not raw_path:
            raise HTTPException(
                status_code=422,
                detail="Missing path (provide 'path' or legacy 'project_path').",
            )

        # Try to obtain or create a session in a tolerant way. Some SessionStore
        # implementations expose 'ensure', others 'get'. If a create method exists
        # prefer that when a session is missing.
        session = None
        try:
            ensure_fn = getattr(sessions, "ensure", None)
            if callable(ensure_fn):
                session = await ensure_fn(sid)
            else:
                get_fn = getattr(sessions, "get", None)
                if callable(get_fn):
                    session = await get_fn(sid)
        except Exception:
            # tolerate and continue to fallback creation below
            session = None

        if session is None:
            # Try to create a session if the store offers a create/new API.
            created = False
            try:
                create_fn = getattr(sessions, "create", None) or getattr(sessions, "new", None)
                if callable(create_fn):
                    maybe = create_fn(sid) if create_fn.__code__.co_argcount >= 1 else create_fn()
                    if inspect.isawaitable(maybe):
                        session = await maybe
                    else:
                        session = maybe
                    created = session is not None
            except Exception:
                session = None

            if not created and session is None:
                # Could not obtain or create a session; instruct the client to re-select.
                logger.warning(
                    "session not found and could not be created",
                    ctx={"session_id": sid},
                )
                return _json_error(
                    project_root=None,
                    where="select",
                    status=400,
                    message=(
                        "Invalid or expired session. Please re-select a project in the UI and try again."
                    ),
                )

        p = Path(raw_path).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Project path does not exist: {p}",
            )

        # Try to match a discovered candidate to enrich metadata
        chosen: Optional[ProjectCandidate] = None
        for c in find_projects(
            p if (p / ".git").exists() else p.parent,
            max_depth=1,
        ):
            if c.path == p:
                chosen = c
                break

        proj_id = _stable_project_id(p)

        if chosen is None:
            selected = ProjectCandidateModel(
                path=str(p),
                score=0,
                kind="unknown",
                markers=[],
                project_id=proj_id,
                children_count=0,
            )
        else:
            selected = ProjectCandidateModel.from_cand(chosen)
            selected.project_id = proj_id  # ensure stable id

        # Persist selection into the session object (support both names)
        try:
            session.project_path = str(p)
        except Exception:
            try:
                setattr(session, "project_path", str(p))
            except Exception:
                pass

        try:
            session.project_root = str(p)
        except Exception:
            try:
                setattr(session, "project_root", str(p))
            except Exception:
                pass

        # Ultra-legacy: some code may still look at session.root
        try:
            session.root = str(p)
        except Exception:
            try:
                setattr(session, "root", str(p))
            except Exception:
                pass

        try:
            meta_map = getattr(session, "meta", None)
            if meta_map is None:
                try:
                    session.meta = {}
                    meta_map = session.meta
                except Exception:
                    # Some session types may not allow attribute assignment; ignore
                    meta_map = None
            if meta_map is not None:
                meta_map.setdefault("project", {}).update(selected.dict())
        except Exception:
            # Best-effort: continue even if meta cannot be updated in-place
            pass

        # Persist session (tolerant to different SessionStore implementations).
        # This must happen BEFORE emitting project_selected to avoid race conditions.
        persisted = False

        async def _maybe_await_call(fn, *args, **kwargs):
            try:
                res = fn(*args, **kwargs)
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:
                raise

        try:
            # Try a broad set of common persistence APIs in order of likelihood.
            for name in ("save", "update", "persist", "put", "set", "store", "write"):
                fn = getattr(sessions, name, None)
                if callable(fn):
                    # Prefer signatures that accept (session) or (id, session)
                    try:
                        # Try (session,) first
                        await _maybe_await_call(fn, session)
                        persisted = True
                        break
                    except TypeError:
                        # try (id, session)
                        sid_val = getattr(session, "id", sid)
                        await _maybe_await_call(fn, sid_val, session)
                        persisted = True
                        break
            # Last resorts: set(id, session) with explicit names
            if not persisted:
                for name in ("set", "put"):
                    fn = getattr(sessions, name, None)
                    if callable(fn):
                        sid_val = getattr(session, "id", sid)
                        await _maybe_await_call(fn, sid_val, session)
                        persisted = True
                        break

            if not persisted:
                # Do not silently proceed: without persistence downstream requests may use
                # a default root (potentially the AI Dev Bot repo). Instruct user to re-select.
                logger.warning(
                    "session persistence API not found; refusing to select project",
                    ctx={"session_id": getattr(session, "id", sid)},
                )
                return _json_error(
                    project_root=None,
                    where="select",
                    status=500,
                    message=(
                        "Session store does not support persistence (save/update/set/put missing); cannot select project."
                        " Please re-select a project in the UI and try again."
                    ),
                )
        except Exception as persist_err:
            logger.warning(
                "session persistence failed; refusing to select project",
                ctx={"session_id": getattr(session, "id", sid), "err": type(persist_err).__name__},
                exc=persist_err,
            )
            return _json_error(
                project_root=None,
                where="select",
                status=500,
                message=(
                    f"session save failed: {type(persist_err).__name__}: {persist_err}."
                    " Please re-select a project in the UI and try again."
                ),
                exc=persist_err,
            )

        # Set session cookie for clients to use in follow-up requests. Keep compatibility
        # with older middleware by also setting the canonical AIDEV_SESSION_COOKIE if present.
        try:
            if response is not None:
                # Primary cookie as chosen for UI selection
                response.set_cookie(
                    "session_id",
                    getattr(session, "id", sid),
                    httponly=True,
                    samesite="lax",
                )
                # Also set legacy/canonical cookie name if configured to reduce mismatch risk
                aidev_cookie_name = os.getenv("AIDEV_SESSION_COOKIE", "aidev_session")
                if aidev_cookie_name and aidev_cookie_name != "session_id":
                    try:
                        response.set_cookie(
                            aidev_cookie_name,
                            getattr(session, "id", sid),
                            httponly=True,
                            samesite="lax",
                        )
                    except Exception:
                        # best-effort
                        pass
        except Exception as cookie_err:
            # Best-effort: cookie failure should not prevent selection.
            logger.warning(
                "failed to set session cookie",
                ctx={"session_id": getattr(session, "id", sid), "err": type(cookie_err).__name__},
                exc=cookie_err,
            )

        ev_project_selected(
            root=str(p),
            session_id=getattr(session, "id", sid),
            project_id=proj_id,
        )
        logger.info(
            "workspace_selected",
            ctx={
                "path": str(p),
                "session_id": getattr(session, "id", sid),
                "project_id": proj_id,
            },
        )

        return SelectResponse(
            session_id=getattr(session, "id", sid),
            selected=selected,
            project_id=proj_id,
        )
    except HTTPException as he:
        return _json_error(
            project_root=None,
            where="select",
            status=he.status_code,
            message=str(he.detail),
            exc=he,
        )
    except Exception as e:
        return _json_error(
            project_root=None,
            where="select",
            status=500,
            message=f"select failed: {type(e).__name__}: {e}",
            exc=e,
        )


@router.get("/workspaces/structure", response_model=StructureResponse)
async def get_structure(
    session_id: str = Query(...),
    include: List[str] = Query(default_factory=list),
    exclude: List[str] = Query(default_factory=list),
    max_context_kb: int = Query(default=1024, ge=64, le=1024 * 64),
    strip_comments: bool = Query(default=False),
    return_map: bool = Query(default=True),
    out_path: Optional[str] = Query(
        default=None,
        description=(
            "Optional output file path; defaults to .aidev/project_map.json. "
            "When using the default, the written file is the lean map from repo_map; "
            "custom paths get the richer cards-based map."
        ),
    ),
    # Optional per-request override of the project root; does NOT persist to session
    project_path: Optional[str] = Query(default=None),
    x_aidev_project: Optional[str] = Header(None, alias="X-AIDEV-PROJECT"),
    sessions: SessionStore = Depends(get_session_store),
) -> StructureResponse | JSONResponse:
    session = await sessions.ensure(session_id)
    project_root: Optional[Path] = None
    try:
        # Resolve per-request override if provided, otherwise use header or session-selected root
        project_root, src = _resolve_request_project_root(session, project_path, x_aidev_project, source_label="query")

        cfg, _cfg_path = load_project_config(project_root, None)
        includes = include or list(cfg.get("discovery", {}).get("includes", []))
        excludes = exclude or list(cfg.get("discovery", {}).get("excludes", []))

        ev_status(
            "structure: start",
            session_id=session.id,
            include=includes,
            exclude=excludes,
            max_context_kb=int(max_context_kb),
            progress_pct=5,
        )

        max_kb = max(1, int(max_context_kb))
        struct, _ctx_blob = discover_structure(
            project_root,
            includes,
            excludes,
            max_total_kb=max_kb,
            strip_comments=bool(strip_comments),
        )
        meta = compact_structure(struct)
        kb = KnowledgeBase(project_root, struct)

        # Best-effort heuristic summaries (non-LLM) to seed cards
        try:
            kb.update_cards()
        except Exception as seed_err:
            logger.warning(
                "heuristic card seed failed (continuing)",
                ctx={"err": type(seed_err).__name__},
                exc=seed_err,
            )

        # Where to write
        if out_path:
            out = Path(out_path)
            if not out.is_absolute():
                out = project_root / out
        else:
            out = project_root / ".aidev" / "project_map.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        # Prefer KB writer; for .aidev/project_map.json this delegates to repo_map
        try:
            path_str = kb.save_project_map(out, project_meta=meta)
            out = Path(path_str)
            payload = (
                json.loads(out.read_text(encoding="utf-8"))
                if return_map
                else None
            )
        except Exception:
            # Very old KB: fallback to a minimal project_map-like payload
            files: List[Dict[str, Any]] = []
            try:
                for card in getattr(kb, "cards", {}).values():  # type: ignore[attr-defined]
                    files.append(
                        {
                            "id": getattr(card, "id", None),
                            "path": getattr(card, "path", None),
                            "title": getattr(card, "title", None),
                            "summary": getattr(card, "summary", None)
                            or getattr(card, "prompt", None),
                        }
                    )
            except Exception:
                files = []
            payload = {
                "version": 1,
                "project_root": str(project_root),
                "meta": meta,
                "files": files,
            }
            out.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        size = out.stat().st_size if out.exists() else 0

        ev_status(
            "structure: done",
            session_id=session.id,
            bytes=int(size),
            path=str(out),
            progress_pct=100,
        )
        ev_done(
            where="structure",
            ok=True,
            session_id=session.id,
            bytes=int(size),
            path=str(out),
        )

        return StructureResponse(
            path=str(out),
            bytes=int(size),
            structure=(payload if return_map else None),
        )
    except HTTPException as he:
        ev_status(
            "HTTP error",
            session_id=session.id,
            event="structure:error",
            trace=_trace_tail(he),
        )
        ev_done(where="structure", ok=False, session_id=session.id)
        return _json_error(
            project_root=project_root,
            where="structure",
            status=he.status_code,
            message=str(he.detail),
            exc=he,
        )
    except Exception as e:
        ev_status(
            "structure:error",
            session_id=session.id,
            message=str(e),
            trace=_trace_tail(e),
        )
        ev_done(where="structure", ok=False, session_id=session.id)
        return _json_error(
            project_root=project_root,
            where="structure",
            status=500,
            message=f"struct failed: {type(e).__name__}: {e}",
            exc=e,
        )


@router.post("/workspaces/refresh-cards", response_model=CardsIndexResponse)
async def refresh_cards(
    body: RefreshCardsRequest,
    x_aidev_project: Optional[str] = Header(None, alias="X-AIDEV-PROJECT"),
    sessions: SessionStore = Depends(get_session_store),
) -> CardsIndexResponse | JSONResponse:
    session = await sessions.get(body.session_id)
    project_root: Optional[Path] = None
    try:
        # Resolve per-request override if provided, otherwise use header or session-selected root
        project_root, src = _resolve_request_project_root(session, getattr(body, "project_path", None), x_aidev_project, source_label="body")

        cfg, _cfg_path = load_project_config(project_root, None)
        includes = list(cfg.get("discovery", {}).get("includes", []))
        excludes = list(cfg.get("discovery", {}).get("excludes", []))

        struct, _ = discover_structure(
            project_root,
            includes,
            excludes,
            max_total_kb=128,
            strip_comments=False,
        )
        kb = KnowledgeBase(project_root, struct)

        baseline_exists = _has_cards_baseline(project_root)
        changed_only_flag = False if body.force or not baseline_exists else True

        ev_status(
            "refresh-cards: start",
            session_id=session.id,
            baseline_exists=baseline_exists,
            changed_only=changed_only_flag,
            force=bool(body.force),
            progress_pct=5,
        )

        kb.update_cards(
            force=bool(body.force),
            changed_only=changed_only_flag,
        )

        ev_status(
            "refresh-cards: done",
            session_id=session.id,
            progress_pct=100,
        )
        ev_done(where="refresh-cards", ok=True, session_id=session.id)

        idx = _read_cards_index(project_root)
        return CardsIndexResponse(cards_index=idx)
    except HTTPException as he:
        ev_status(
            "refresh-cards:error",
            session_id=session.id,
            message="HTTP error",
            trace=_trace_tail(he),
        )
        ev_done(where="refresh-cards", ok=False, session_id=session.id)
        return _json_error(
            project_root=project_root,
            where="refresh-cards",
            status=he.status_code,
            message=str(he.detail),
            exc=he,
        )
    except Exception as e:
        ev_status(
            "refresh-cards:error",
            session_id=session.id,
            message=str(e),
            trace=_trace_tail(e),
        )
        ev_done(where="refresh-cards", ok=False, session_id=session.id)
        return _json_error(
            project_root=project_root,
            where="refresh-cards",
            status=500,
            message=f"refresh-cards failed: {type(e).__name__}: {e}",
            exc=e,
        )


# -------------------- Unified endpoint -----------------------------------

@router.post("/workspaces/ai/cards", response_model=AICardsResponse)
async def ai_cards(
    body: AICardsRequest,
    x_aidev_project: Optional[str] = Header(None, alias="X-AIDEV-PROJECT"),
    sessions: SessionStore = Depends(get_session_store),
) -> AICardsResponse | JSONResponse:
    """
    Orchestrated endpoint:
      - mode="changed": incremental summarize; enrich only if body.enrich=True
      - mode="deep": incremental summarize; then enrich (project-wide)
      - mode="full": full summarize; then enrich if body.enrich=True

    All summaries/enrichment operate on .aidev/cards/index.json and per-file .card.json files.
    Enrichment never overwrites fresh ai_summary when protect_llm=True.
    """
    session = await sessions.get(body.session_id)
    project_root: Optional[Path] = None
    try:
        # Resolve per-request override if provided, otherwise use header or session-selected root
        project_root, src = _resolve_request_project_root(session, getattr(body, "project_path", None), x_aidev_project, source_label="body")

        # infer enrich default from mode if not explicitly provided
        do_enrich = (
            body.enrich
            if body.enrich is not None
            else (body.mode in ("deep", "full"))
        )
        idx, summarized_count, skipped_count, enriched = _orchestrate_ai_cards(
            project_root=project_root,
            session_id=session.id,
            mode=body.mode,
            model=body.model,
            ttl_days=int(body.ttl_days),
            protect_llm=bool(body.protect_llm),
            do_enrich=bool(do_enrich),
            enrich_top_k=int(body.enrich_top_k),
            focus=body.focus,
        )
        ev_done(where="ai/cards", ok=True, session_id=session.id)
        return AICardsResponse(
            cards_index=idx,
            summarized_count=summarized_count,
            skipped_count=skipped_count,
            enriched=enriched,
        )
    except HTTPException as he:
        ev_status(
            "ai/cards:error",
            session_id=session.id,
            message="HTTP error",
            trace=_trace_tail(he),
        )
        ev_done(where="ai/cards", ok=False, session_id=session.id)
        return _json_error(
            project_root=project_root,
            where="ai/cards",
            status=he.status_code,
            message=str(he.detail),
            exc=he,
        )
    except Exception as e:
        ev_status(
            "ai/cards:error",
            session_id=session.id,
            message=str(e),
            trace=_trace_tail(e),
        )
        ev_done(where="ai/cards", ok=False, session_id=session.id)
        return _json_error(
            project_root=project_root,
            where="ai/cards",
            status=500,
            message=f"ai/cards failed: {type(e).__name__}: {e}",
            exc=e,
        )


# ---- Single card fetch (normalized) -----------------------------------------

class SingleCardResponse(BaseModel):
    path: str
    card: Dict[str, Any]


@router.get("/workspaces/card", response_model=SingleCardResponse)
async def get_card(
    session_id: str = Query(...),
    path: str = Query(
        ...,
        description=(
            "File path relative to the project root (or absolute inside the project)."
        ),
    ),
    # Optional per-request override of the project root; does NOT persist to session
    project_path: Optional[str] = Query(default=None),
    x_aidev_project: Optional[str] = Header(None, alias="X-AIDEV-PROJECT"),
    sessions: SessionStore = Depends(get_session_store),
) -> SingleCardResponse | JSONResponse:
    """
    Return one normalized .card.json for 'path', exposing `card['summary_text']`
    which prefers summary.ai_text, falling back to summary.heuristic.
    """
    session = await sessions.ensure(session_id)
    project_root: Optional[Path] = None
    try:
        # Resolve per-request override if provided, otherwise use header or session-selected root
        project_root, src = _resolve_request_project_root(session, project_path, x_aidev_project, source_label="query")

        # Normalize input path to a project-relative POSIX path
        raw = Path(path)
        if raw.is_absolute():
            try:
                rel = raw.resolve().relative_to(
                    project_root.resolve()
                ).as_posix()
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Path must be inside the selected project.",
                )
        else:
            rel = raw.as_posix()

        card = _read_single_card(project_root, rel)
        if not card:
            raise HTTPException(
                status_code=404,
                detail=f"No card found for: {rel}",
            )

        return SingleCardResponse(path=rel, card=card)

    except HTTPException as he:
        return _json_error(
            project_root=project_root,
            where="card",
            status=he.status_code,
            message=str(he.detail),
            exc=he,
        )
    except Exception as e:
        return _json_error(
            project_root=project_root,
            where="card",
            status=500,
            message=f"card failed: {type(e).__name__}: {e}",
            exc=e,
        )


@router.get("/workspaces/_echo")
async def workspaces_echo(
    session_id: str,
    message: str = "structure: start",
    where: str | None = None,
):
    # Tell the UI “something started…”
    ev_status(message, session_id=session_id, progress_pct=5)
    # …and immediately mark it done under the same topic.
    topic = (
        where
        or (
            message.split(":", 1)[0].strip()
            if ":" in message
            else None
        )
    )
    ev_done(ok=True, where=topic, session_id=session_id)
    ev_status(
        f"{topic}: done" if topic else "done",
        session_id=session_id,
        progress_pct=100,
    )
    return {"ok": True, "sent": {"status": message, "where": topic}}