# aidev/context/brief.py
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

# Optional LLM + logger dependencies
try:
    from ..llm_client import LLMClient, compile_project_brief as _client_compile_project_brief  # type: ignore
except Exception:
    LLMClient = None  # type: ignore
    _client_compile_project_brief = None  # type: ignore

try:
    from .. import logger  # type: ignore
except Exception:
    import logging as _logging

    _log = _logging.getLogger("aidev.context.brief")

    class _Shim:
        def info(self, msg, ctx=None): _log.info(msg)
        def warn(self, msg, ctx=None, exc=None): _log.warning(msg)
        def warning(self, msg, ctx=None, exc=None): _log.warning(msg)
        def error(self, msg, ctx=None, exc=None): _log.error(msg)
        def exception(self, msg, ctx=None): _log.exception(msg)

    logger = _Shim()

# -----------------------
# Canonicalization & hash
# -----------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _to_canonical(obj: Any) -> Any:
    """
    Convert arbitrary brief input (str/dict/list) into a stable, JSON-serializable shape.
    - strings -> trimmed, single-spaced
    - dict -> {sorted keys: canonical(value)}
    - list/tuple -> [canonical(items)]
    - other scalars -> as-is
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        s = _WHITESPACE_RE.sub(" ", s)
        return s
    if isinstance(obj, Mapping):
        return {k: _to_canonical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_to_canonical(x) for x in obj]
    return obj


def canonicalize_brief(brief: Any) -> Any:
    """
    Ensure the brief is in a canonical shape for hashing.
    If it's a string, returns a normalized string.
    If it's mapping-like, returns a sorted/normalized dict.
    """
    c = _to_canonical(brief)
    return c  # may be str or dict


def compute_brief_hash(brief: Any) -> str:
    """
    Stable content hash (sha256) of the canonical brief.
    """
    canon = canonicalize_brief(brief)
    if isinstance(canon, str):
        blob = canon
    else:
        blob = json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -----------------------
# Brief discovery/builder
# -----------------------

# Canonical locations under .aidev/ for new projects.
APP_DESCRIP_PATH = ".aidev/app_descrip.txt"
PROJECT_DESCRIPTION_PATH = ".aidev/project_description.md"
PROJECT_METADATA_PATH = ".aidev/project_metadata.json"

# When we fall back to "existing brief-like files", prefer structured brief first.
BRIEF_CANDIDATE_FILES: tuple[str, ...] = (
    PROJECT_DESCRIPTION_PATH,      # .aidev/project_description.md
    ".aidev/brief.md",
    "README.md",
    APP_DESCRIP_PATH,             # .aidev/app_descrip.txt
)

PROJECT_MAP_PATH = ".aidev/project_map.json"
DEFAULT_BRIEF_OUT = ".aidev/brief.md"
MAX_BRIEF_CHARS = 4000
MAX_CARD_SNIPPETS = 24  # cap how many files we sample into the brief

# Primary cards index candidates: prefer .aidev/cards/index.json (v2), then legacy.
CARDS_INDEX_CANDIDATES: tuple[str, ...] = (
    ".aidev/cards/index.json",  # canonical v2 cards index
    ".aidev/cards.json",        # legacy
)


def _read_first_existing(root: Path, names: Iterable[str]) -> tuple[str, str]:
    """
    Returns (text, path_str) for the first existing non-empty file among names.
    Empty text if none found.
    """
    for name in names:
        p = (root / name).resolve()
        try:
            if p.is_file():
                txt = p.read_text(encoding="utf-8", errors="replace")
                if txt.strip():
                    return txt, str(p)
        except Exception:
            # ignore read errors, try next
            pass
    return "", ""


def _load_project_map(root: Path) -> dict:
    """
    Load .aidev/project_map.json if present. This is a compact, LLM-facing map,
    not the full cards index. We use it only as a fallback when no cards index
    is available.
    """
    p = (root / PROJECT_MAP_PATH).resolve()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_json_load(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_cards_index(raw: Any) -> Dict[str, Any]:
    """
    Accept several shapes and normalize to { rel_path: node_dict }.

    Supported shapes:
    - { "<path>": { ... }, ... }  # flat map (current .aidev/cards/index.json or .aidev/cards.json)
    - { "cards": { "<path>": { ... } } }
    - { "files": [ { "path": \"...\", ... }, ... ] }
    """
    if isinstance(raw, dict):
        if "cards" in raw and isinstance(raw["cards"], dict):
            return raw["cards"]
        if "files" in raw and isinstance(raw["files"], list):
            out: Dict[str, Any] = {}
            for node in raw["files"]:
                if isinstance(node, dict) and "path" in node:
                    out[str(node["path"])] = node
            return out
        return raw
    return {}


def _load_cards_index(root: Path) -> Dict[str, Any]:
    """
    Load the canonical cards index, preferring .aidev/cards/index.json (v2),
    then legacy fallbacks.
    """
    for rel in CARDS_INDEX_CANDIDATES:
        p = (root / rel).resolve()
        if not p.is_file():
            continue
        raw = _safe_json_load(p)
        if raw is None:
            continue
        idx = _normalize_cards_index(raw)
        if isinstance(idx, dict) and idx:
            return idx
    return {}


def _clip_line(s: str, n: int = 320) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _extract_card_text(node: dict) -> str | None:
    """
    Prefer AI summary fields; fall back to heuristic node fields.

    Supports:
    - New .aidev/index.json shape:
        node["summary"] = { "heuristic": "...", "ai_text": "...", ... }
        node["ai_summary"] = "<string>" (optional)
    - Older shapes where summary/ai_summary may be dicts/strings.
    """
    if not isinstance(node, dict):
        return None

    # 1) New-style summary dict with heuristic/ai_text fields.
    summary = node.get("summary")
    if isinstance(summary, dict):
        for key in ("ai_text", "heuristic", "summary", "short", "text", "content"):
            v = summary.get(key)
            if isinstance(v, str) and v.strip():
                return _clip_line(v)
    elif isinstance(summary, str) and summary.strip():
        # In case of older flat textual summaries.
        return _clip_line(summary)

    # 2) ai_summary variants (string or dict).
    ai = node.get("ai_summary")
    if isinstance(ai, str) and ai.strip():
        return _clip_line(ai)
    if isinstance(ai, dict):
        for k in ("summary", "short", "text", "content", "ai_summary"):
            v = ai.get(k)
            if isinstance(v, str) and v.strip():
                return _clip_line(v)

    # 3) Other top-level textual fallbacks.
    for k in ("description", "desc", "notes"):
        v = node.get(k)
        if isinstance(v, str) and v.strip():
            return _clip_line(v)

    return None


def _build_brief_from_cards(root: Path) -> str:
    """
    Construct a compact project brief from the cards index (preferred) or,
    as a fallback, from the project map.

    We sample up to MAX_CARD_SNIPPETS entries and produce a readable outline
    that LLM prompts can consume.
    """
    cards = _load_cards_index(root)
    items: list[tuple[str, dict]] = []

    if cards:
        # Deterministic sample: sort by path and take the first N.
        for path, node in sorted(cards.items(), key=lambda kv: kv[0])[:MAX_CARD_SNIPPETS]:
            if isinstance(node, dict):
                items.append((path, node))
    else:
        # Fallback: use the project_map.json file list if present.
        pm = _load_project_map(root)
        files = pm.get("files")
        if isinstance(files, list):
            for node in files[:MAX_CARD_SNIPPETS]:
                if not isinstance(node, dict):
                    continue
                path = str(node.get("path", "") or "")
                if not path:
                    path = "<unknown>"
                items.append((path, node))

    lines: list[str] = []
    lines.append("# Project Brief (auto-built)\n")
    lines.append("This brief is auto-constructed from the current project cards/map.\n")

    for path, node in items:
        text = _extract_card_text(node)
        if not text:
            continue
        lines.append(f"\n## {path}\n")
        lines.append(text)

    brief = "\n".join(lines).strip()
    if not brief:
        brief = (
            "# Project Brief\n"
            "(No cards or project map summaries available yet. "
            "Generate the project map/cards and retry.)"
        )
    return brief


def _clip_total(s: str, n: int = MAX_BRIEF_CHARS) -> str:
    s = s.replace("\r\n", "\n")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _write_out(root: Path, text: str, out_rel: str = DEFAULT_BRIEF_OUT) -> str:
    """
    Write a brief-like file (typically .aidev/brief.md), clipping to MAX_BRIEF_CHARS.
    Returns the absolute path as a string.
    """
    out = (root / out_rel).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    safe = _clip_total(text, MAX_BRIEF_CHARS)
    out.write_text(safe, encoding="utf-8")
    return str(out)


# -----------------------
# Repo-level hints from index/map
# -----------------------

def _summarize_repo_from_cards(root: Path) -> dict:
    """
    Infer high-level repo hints (languages, frameworks, tools) from the cards
    index (preferred) or project map. This is intentionally small and
    token-cheap so it can be passed into the LLM brief compiler.
    """
    cards = _load_cards_index(root)
    from_project_map = False

    if not cards:
        pm = _load_project_map(root)
        files = pm.get("files")
        if isinstance(files, list):
            cards = {}
            for node in files:
                if isinstance(node, dict) and "path" in node:
                    cards[str(node["path"])] = node
            from_project_map = True

    if not cards:
        return {}

    lang_counts: Dict[str, int] = {}
    frameworks: set[str] = set()
    tools: set[str] = set()

    # Pre-compute basenames + paths for tool/framework detection.
    paths = list(cards.keys())
    basenames = {Path(p).name.lower() for p in paths}

    # Language display normalization
    def _norm_lang(raw: str) -> str:
        r = raw.lower().strip()
        mapping = {
            "python": "Python",
            "py": "Python",
            "typescript": "TypeScript",
            "ts": "TypeScript",
            "tsx": "TypeScript/React",
            "javascript": "JavaScript",
            "js": "JavaScript",
            "jsx": "JavaScript/React",
            "mjs": "JavaScript",
            "cjs": "JavaScript",
            "php": "PHP",
            "dart": "Dart",
            "html": "HTML",
            "css": "CSS",
            "json": "JSON",
            "markdown": "Markdown",
            "md": "Markdown",
            "text": "Text",
        }
        return mapping.get(r, raw)

    # Pass 1: language counts based on node["lang"] / node["language"] / extension.
    for path, node in cards.items():
        lang = None
        if isinstance(node, dict):
            lang = node.get("lang") or node.get("language")
        if not isinstance(lang, str) or not lang.strip():
            ext = Path(path).suffix.lower()
            ext_map = {
                ".py": "python",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".js": "javascript",
                ".jsx": "javascript",
                ".mjs": "javascript",
                ".cjs": "javascript",
                ".php": "php",
                ".dart": "dart",
                ".html": "html",
                ".htm": "html",
                ".css": "css",
                ".json": "json",
                ".md": "markdown",
            }
            lang = ext_map.get(ext)
        if not lang:
            continue
        disp = _norm_lang(lang)
        lang_counts[disp] = lang_counts.get(disp, 0) + 1

    # Config / tool detection by filenames.
    TOOL_BY_FILENAME = {
        "pyproject.toml": "Python project (pyproject.toml)",
        "requirements.txt": "Python dependencies (requirements.txt)",
        "poetry.lock": "Poetry",
        "package.json": "Node.js (package.json)",
        "composer.json": "Composer (PHP)",
        ".eslintrc": "ESLint",
        ".eslintrc.js": "ESLint",
        ".eslintrc.cjs": "ESLint",
        "eslint.config.js": "ESLint",
        "eslint.config.mjs": "ESLint",
        ".prettierrc": "Prettier",
        ".prettierrc.js": "Prettier",
        ".prettierrc.cjs": "Prettier",
        "prettier.config.js": "Prettier",
        "ruff.toml": "Ruff",
        "setup.cfg": "Python configuration (setup.cfg)",
        "tox.ini": "tox",
        "pytest.ini": "pytest",
        "tsconfig.json": "TypeScript (tsconfig.json)",
        "pubspec.yaml": "Dart/Flutter (pubspec.yaml)",
    }

    for name in basenames:
        tool = TOOL_BY_FILENAME.get(name)
        if tool:
            tools.add(tool)

    # GitHub Actions (directory-based).
    if any(p.startswith(".github/workflows/") for p in paths):
        tools.add("GitHub Actions")

    # Basic framework heuristics from filenames/paths.
    if "pubspec.yaml" in basenames or any(p.endswith(".dart") for p in paths):
        frameworks.add("Flutter/Dart")

    if any(Path(p).name.lower() in ("next.config.js", "next.config.mjs", "next.config.ts") for p in paths):
        frameworks.add("Next.js")

    if any(
        ("/app/" in p or "/pages/" in p) and (p.endswith(".tsx") or p.endswith(".jsx"))
        for p in paths
    ):
        frameworks.add("React")

    if "composer.json" in basenames:
        frameworks.add("PHP (Composer-based)")

    languages_sorted = sorted(lang_counts.keys(), key=str.lower)
    primary_language = None
    if lang_counts:
        primary_language = max(lang_counts.items(), key=lambda kv: kv[1])[0]

    # Hard-cap list sizes to keep token costs low.
    def _cap_list(xs: Iterable[str], k: int = 8) -> list[str]:
        out: list[str] = []
        for x in xs:
            if x in out:
                continue
            out.append(x)
            if len(out) >= k:
                break
        return out

    return {
        "source": "project_map" if from_project_map else "cards_index",
        "primary_language": primary_language,
        "languages": _cap_list(languages_sorted),
        "language_counts": {k: lang_counts[k] for k in languages_sorted},
        "frameworks": _cap_list(sorted(frameworks, key=str.lower)),
        "tools": _cap_list(sorted(tools, key=str.lower)),
    }


def _repo_hints_for_llm(root: Path) -> dict:
    try:
        return _summarize_repo_from_cards(root) or {}
    except Exception as e:
        try:
            logger.warning("brief: repo hint extraction failed", ctx={"err": str(e)})
        except Exception:
            pass
        return {}


# -----------------------
# app_descrip → compiled brief
# -----------------------

def _load_metadata(path: Path) -> dict:
    """
    Load project metadata JSON from the canonical .aidev location.
    Legacy root-level project_metadata.json is no longer supported.
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_metadata(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _infer_project_name(app_text: str, root: Path) -> str:
    """
    Best-effort project name:
    - First markdown heading line starting with '#' if present.
    - Else first non-empty line.
    - Else directory name.
    """
    for line in app_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # strip leading '#' and whitespace
            return stripped.lstrip("#").strip() or root.name
        return stripped
    return root.name


def _infer_short_tagline(app_text: str) -> str:
    """
    A short tagline derived from the description. We just use the
    first couple hundred characters, normalized.
    """
    return _clip_line(app_text, n=160)


def _llm_compile_brief(app_text: str, root: Path, existing_meta: dict) -> tuple[str, dict]:
    """
    LLM-backed brief compiler.

    Delegates to the shared llm_client.compile_project_brief helper so we
    reuse the central brief_compile prompt + schema.

    On any failure or missing API keys, returns ("", existing_meta) so the caller
    can fall back to the deterministic implementation.
    """
    if _client_compile_project_brief is None:
        return "", existing_meta

    # Build an augmented description that keeps app_descrip as the primary source
    # of truth but also threads through any existing metadata and repo hints.
    repo_hints = _repo_hints_for_llm(root)
    try:
        repo_hints_json = json.dumps(repo_hints or {}, ensure_ascii=False, indent=2)
    except Exception:
        repo_hints_json = "{}"
    try:
        meta_json = json.dumps(existing_meta or {}, ensure_ascii=False, indent=2)
    except Exception:
        meta_json = "{}"

    augmented_app_text = (
        (app_text or "").strip()
        + "\n\n---\n\n"
        "Additional context for the brief compiler (JSON; optional, may be partial):\n"
        f"existing_metadata = {meta_json}\n"
        f"repo_hints = {repo_hints_json}\n"
    )

    try:
        md_body, model_meta, _resp = _client_compile_project_brief(augmented_app_text)
    except Exception as e:
        try:
            logger.warning(
                "brief: LLM compile via shared helper failed; using deterministic brief",
                ctx={"err": str(e)},
            )
        except Exception:
            pass
        return "", existing_meta

    if not isinstance(model_meta, dict):
        model_meta = {}

    md_body = (md_body or "").strip()
    if not md_body:
        return "", existing_meta

    # Merge metadata: model keys override existing ones, then ensure core fields.
    meta: dict = dict(existing_meta)
    meta.update(model_meta)

    if not meta.get("project_name"):
        meta["project_name"] = _infer_project_name(app_text, root)
    if not meta.get("short_tagline"):
        meta["short_tagline"] = _infer_short_tagline(app_text)

    # Ensure we have an H1 heading at the top.
    lines = md_body.splitlines()
    first_nonempty_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_nonempty_idx = i
            break

    if first_nonempty_idx is None:
        md_body_lines: list[str] = [
            f"# {meta['project_name']}",
            "",
            "## Overview",
            "",
            app_text.strip() or "(No description provided yet.)",
        ]
        md_body = "\n".join(md_body_lines)
    else:
        first = lines[first_nonempty_idx]
        if not first.lstrip().startswith("#"):
            heading = f"# {meta['project_name']}"
            md_body = "\n".join(
                lines[:first_nonempty_idx] + [heading, ""] + lines[first_nonempty_idx:]
            )

    md_full = "\n" + md_body.strip() + "\n"
    return md_full, meta


def _compile_brief_from_app_descrip_deterministic(
    app_text: str, root: Path, existing_meta: dict | None = None
) -> tuple[str, dict]:
    """
    Deterministically compile app_descrip.txt into:
    - project_description.md (Markdown string)
    - project_metadata (dict, without hash/timestamp which are added by caller)

    This is the LLM-free fallback implementation. It is used when the LLM is
    unavailable or returns an invalid/empty result.
    """
    existing_meta = existing_meta or {}

    project_name = existing_meta.get("project_name") or _infer_project_name(app_text, root)
    short_tagline = existing_meta.get("short_tagline") or _infer_short_tagline(app_text)

    lines: list[str] = []

    lines.append(f"# {project_name}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    # Keep the original text as-is in the overview; we can get fancier later.
    lines.append(app_text.strip() or "(No description provided yet.)")

    md = "\n".join(lines).rstrip() + "\n"

    meta: dict = {
        "project_name": project_name,
        "short_tagline": short_tagline,
    }
    # Preserve any other existing keys the caller might have set.
    for k, v in existing_meta.items():
        if k not in meta:
            meta[k] = v

    return md, meta


def _compile_brief_from_app_descrip(
    app_text: str, root: Path, existing_meta: dict | None = None
) -> tuple[str, dict]:
    """
    Compile app_descrip.txt into:
    - project_description.md (Markdown string)
    - project_metadata (dict, without hash/timestamp which are added by caller)

    Behavior:
    - First, tries to use the LLM (via _llm_compile_brief) to generate a richer
      brief + metadata.
    - If the LLM is unavailable or returns an empty/invalid result, falls back
      to the deterministic implementation.
    """
    existing_meta = existing_meta or {}

    # 1) Try LLM-backed compiler.
    md_llm, meta_llm = _llm_compile_brief(app_text, root, existing_meta)
    if md_llm.strip():
        return md_llm, meta_llm

    # 2) Fallback: deterministic behavior.
    return _compile_brief_from_app_descrip_deterministic(app_text, root, existing_meta)


def _ensure_compiled_brief(
    root: Path,
    *,
    force_refresh: bool = False,
    ttl_hours: float | None = None,
) -> tuple[str, str]:
    """
    Ensure that, if an app description exists, we have a fresh compiled brief.

    Canonical source of truth is .aidev/app_descrip.txt.

    Steps:
    - Reads .aidev/app_descrip.txt.
    - Computes its hash.
    - Loads .aidev/project_metadata.json and compares last_brief_source_hash.
    - If missing / mismatched / stale (or force_refresh=True), regenerates:
        - .aidev/project_description.md
        - .aidev/project_metadata.json
      and updates hash + last_updated.
    - Returns (brief_text, brief_path) for project_description.md if available.
      Otherwise ("", "").
    """
    # Locate app_descrip.txt
    app_candidates = [
        (root / APP_DESCRIP_PATH).resolve(),      # .aidev/app_descrip.txt
    ]
    app_path: Path | None = None
    for candidate in app_candidates:
        if candidate.is_file():
            app_path = candidate
            break

    if app_path is None:
        return "", ""

    try:
        app_text = app_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", ""

    if not app_text.strip():
        return "", ""

    current_hash = compute_brief_hash(app_text)

    meta_path = (root / PROJECT_METADATA_PATH).resolve()
    desc_path = (root / PROJECT_DESCRIPTION_PATH).resolve()

    meta = _load_metadata(meta_path)
    last_hash = meta.get("last_brief_source_hash")
    last_updated = meta.get("last_updated")

    need_regen = force_refresh or (not desc_path.is_file()) or (last_hash != current_hash)

    # TTL-based refresh (optional)
    if not need_regen and ttl_hours is not None and last_updated:
        try:
            # Accept both Z and offset forms.
            ts = last_updated.rstrip("Z")
            updated_dt = datetime.fromisoformat(ts)
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - updated_dt).total_seconds() / 3600.0
            if age_hours > ttl_hours:
                need_regen = True
        except Exception:
            # If parsing fails, be conservative and refresh.
            need_regen = True

    if need_regen:
        md_text, meta_obj = _compile_brief_from_app_descrip(app_text, root, existing_meta=meta)
        meta_obj["last_brief_source_hash"] = current_hash
        meta_obj["last_updated"] = _utc_now_iso()

        # Write files
        desc_path.parent.mkdir(parents=True, exist_ok=True)
        desc_path.write_text(md_text, encoding="utf-8")

        _save_metadata(meta_path, meta_obj)

        # Return the full text; callers can apply any clipping policy they want.
        return md_text, str(desc_path)

    # No regen needed; try to read existing project_description.md
    try:
        existing = desc_path.read_text(encoding="utf-8", errors="replace")
        if existing.strip():
            return existing, str(desc_path)
    except Exception:
        # If we can't read it for some reason, fall back to regeneration once.
        try:
            md_text, meta_obj = _compile_brief_from_app_descrip(app_text, root, existing_meta=meta)
            meta_obj["last_brief_source_hash"] = current_hash
            meta_obj["last_updated"] = _utc_now_iso()
            desc_path.parent.mkdir(parents=True, exist_ok=True)
            desc_path.write_text(md_text, encoding="utf-8")
            _save_metadata(meta_path, meta_obj)
            return md_text, str(desc_path)
        except Exception:
            return "", ""

    return "", ""


# -----------------------
# Public entrypoint
# -----------------------

def get_or_build(
    project_root: str | Path,
    create_if_missing: bool = True,
    *,
    model: str | None = None,
    force: bool | None = None,
    force_refresh: bool | None = None,
    ttl_hours: float | None = None,
) -> Dict[str, Any]:
    """
    Returns a structured brief descriptor:

        {
            "text": <brief_markdown>,
            "hash": <stable_sha256_of_brief>,
            "path": <source_file_path>,
        }

    New behavior:
    - If an app description exists, we treat it as the human source of truth
      (.aidev/app_descrip.txt).
    - We compile it into .aidev/project_description.md + .aidev/project_metadata.json
      (hash-based; only when it changes or we force/TTL-refresh).
      The compilation uses the LLM by default and falls back to a deterministic
      implementation if the LLM is unavailable.
    - We then prefer the compiled .aidev/project_description.md as the brief.
    - If no compiled brief is available, we fall back to existing files:
      .aidev/project_description.md → .aidev/brief.md → README.md → .aidev/app_descrip.txt.
    - If still nothing is available and create_if_missing is True, we build
      a brief from cards/map and write it to .aidev/brief.md.

    Notes:
    - `model` is currently accepted for API compatibility but not used; it is
      reserved for future model-aware compilation behavior.
    - `force` and `force_refresh` are aliases; `force` from the Orchestrator
      takes precedence if provided.
    """
    root = Path(project_root).resolve()

    # Normalize the force flag: Orchestrator passes `force=...`; keep
    # `force_refresh` as an optional alias for any other callers.
    if force is not None:
        effective_force = bool(force)
    elif force_refresh is not None:
        effective_force = bool(force_refresh)
    else:
        effective_force = False

    # 0) If the user has an app description, ensure we have a compiled brief
    #    and prefer that as the PROJECT_BRIEF.
    text, path_str = _ensure_compiled_brief(
        root,
        force_refresh=effective_force,
        ttl_hours=ttl_hours,
        # `model` can be threaded into the compiler if/when needed.
    )
    if text.strip():
        brief_text = _clip_total(text)
        return {
            "text": brief_text,
            "hash": compute_brief_hash(brief_text),
            "path": path_str,
        }

    # 1) Try existing files (structured brief, legacy brief, README, raw app_descrip).
    text, path_str = _read_first_existing(root, BRIEF_CANDIDATE_FILES)
    if text.strip():
        brief_text = _clip_total(text)
        return {
            "text": brief_text,
            "hash": compute_brief_hash(brief_text),
            "path": path_str,
        }

    # 2) Build from cards/map if requested.
    if create_if_missing:
        text = _build_brief_from_cards(root)
        out_path = _write_out(root, text, DEFAULT_BRIEF_OUT)
        brief_text = _clip_total(text)
        return {
            "text": brief_text,
            "hash": compute_brief_hash(brief_text),
            "path": str(out_path),
        }

    # 3) Nothing available.
    return {
        "text": "",
        "hash": "",
        "path": "",
    }


__all__ = [
    "get_or_build",
    "canonicalize_brief",
    "compute_brief_hash",
]
