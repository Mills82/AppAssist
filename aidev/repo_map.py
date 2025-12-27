# aidev/repo_map.py
"""
Build and cache a compact, LLM-friendly project map from existing cards.

- Reads .aidev/cards/index.json (and a few legacy fallbacks).
- Produces .aidev/project_map.json (token-lean for LLM context).
- Only fields kept / added:

  Top-level:
    {
      "version": 1,
      "generated_at": ISO-UTC,
      "root": "<absolute path to project_root>",
      "total_files": <int>,
      "language_kinds": { "<language>": <count>, ... },
      "by_ext": { "<.ext>": <count>, ... },
      "by_top": { "<top-level dir or file>": <count>, ... },
      "files": [ ... ]
    }

  Per-file:
    {
      "path": "aidev/repo_map.py",
      "language": "python",
      "kind": "code" | "test" | "config" | "doc" | "ui" | "asset" | "other",
      "summary": "...",
      # Optional hints (small, token-lean):
      "routes": [...],
      "cli_args": [...],
      "env_vars": [...],
      "public_api": [...],
      "changed": true,  # optional, only when staleness.changed is true
    }

No repo_map.json is written; only .aidev/project_map.json.

Extra helper:
- summarize_repo_from_map(project_root) -> {
      "languages": [...],
      "frameworks": [...],
      "tools": [...],
      "file_count": int,
  }
  for feeding repo-level hints into the brief compiler or other LLM calls.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------
# Config / caps
# ----------------------------
PROJECT_MAP_REL = Path(".aidev/project_map.json")

# Keep LLM payload small but useful
SUMMARY_MAX_CHARS = 500
HINT_KEYS = ("routes", "cli_args", "env_vars", "public_api")

# Cap the number of file entries emitted into .aidev/project_map.json.
# This is a safety valve for downstream prompt payload budgets.
PROJECT_MAP_MAX_FILES: Optional[int] = None

# Language detection by extension (fallback if card has no language/lang)
PY_EXTS = {".py"}
TS_EXTS = {".ts", ".tsx"}
JS_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
JSON_EXTS = {".json"}
MARKUP_EXTS = {".md", ".rst", ".adoc"}
STYLE_EXTS = {".css", ".scss", ".sass"}
HTML_EXTS = {".html", ".htm"}

# Skip these directories entirely
SKIP_DIRS = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "out",
    ".idea",
    ".vscode",
    "__pycache__",
    ".aidev",  # exclude .aidev content from map
}

# Skip noisy basenames/suffixes
SKIP_BASENAMES = {
    "app.log",
    ".env",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".prettierrc",
    ".npmrc",
    ".dockerignore",
    ".aidev.json",
    "env",
}
SKIP_SUFFIXES = {
    ".log",
    ".bin",
    ".sqlite",
    ".db",
    ".bak",
    ".tmp",
    ".lock",
}

# ----------------------------
# Utilities
# ----------------------------

def _read_text_safe(p: Path, max_bytes: int = 200_000) -> str:
    """
    Best-effort text read with a byte cap and tolerant decoding.
    Used only for heuristic summaries when no AI summary is available.
    """
    try:
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    except Exception:
        return ""


def _lang_for_path(p: Path) -> str:
    """
    Fallback language detection by extension. Primary language comes from cards
    when available (card['language'] or card['lang']).
    """
    ext = p.suffix.lower()
    if ext in PY_EXTS:
        return "python"
    if ext in TS_EXTS:
        return "typescript"
    if ext in JS_EXTS:
        return "javascript"
    if ext in JSON_EXTS:
        return "json"
    if ext in HTML_EXTS:
        return "html"
    if ext in STYLE_EXTS:
        return "css"
    if ext == ".md":
        return "markdown"
    if ext in MARKUP_EXTS:
        return "text"
    return "text"


def _kind_for_path(rel_path: str, language: str) -> str:
    """
    Coarse structural kind for the file, used by edit_context to prioritize
    neighbors. We keep this deliberately simple and stable.

    Possible values:
      - "code"
      - "test"
      - "config"
      - "doc"
      - "ui"
      - "asset"
      - "other"
    """
    path_lower = rel_path.lower()
    p = Path(rel_path)

    # Tests
    name = p.name.lower()
    parent_parts = [part.lower() for part in p.parts]

    if (
        "tests" in parent_parts
        or "test" in parent_parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".spec.ts")
        or name.endswith(".test.ts")
    ):
        return "test"

    # Config / infra
    if any(
        key in path_lower
        for key in (
            "pyproject.toml",
            "requirements.txt",
            "setup.cfg",
            "setup.py",
            "tsconfig",
            "eslint",
            "prettier",
            "dockerfile",
            "docker-compose",
            "config",
            ".env",
        )
    ) or Path(rel_path).suffix.lower() in {".toml", ".ini", ".yaml", ".yml"}:
        return "config"

    # Docs
    if Path(rel_path).suffix.lower() in {".md", ".rst", ".adoc"} or any(
        k in path_lower for k in ("docs/", "/docs/", "readme")
    ):
        return "doc"

    # UI-ish
    if language in {"javascript", "typescript"} and any(
        k in path_lower for k in ("ui/", "components/", "pages/", "routes/")
    ):
        return "ui"
    if Path(rel_path).suffix.lower() in {".html", ".htm"}:
        return "ui"
    if Path(rel_path).suffix.lower() in STYLE_EXTS:
        return "ui"

    # Assets
    if Path(rel_path).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".svg",
        ".mp4",
        ".mov",
        ".mp3",
        ".wav",
    }:
        return "asset"

    # Code default
    if language in {"python", "javascript", "typescript"}:
        return "code"

    # Fallback
    return "other"


def _is_entrypoint(rel_path: str, lang: str) -> bool:
    """
    Heuristic: is this file likely to be a main entrypoint / root script?
    """
    p = Path(rel_path)
    name = p.name.lower()
    parts = [part.lower() for part in p.parts]

    if lang == "python":
        if name in {"main.py", "app.py", "server.py"}:
            return True
        # e.g. src/aidev_cli.py, manage.py, etc.
        if name.endswith("_cli.py") or name == "manage.py":
            return True

    if lang in {"javascript", "typescript"}:
        if name in {"index.tsx", "index.ts", "index.js", "app.tsx", "app.jsx"}:
            return True
        if "pages" in parts or "routes" in parts:
            # Only treat top-level layout/entry as entrypoint-ish
            if name in {"_app.tsx", "_app.js", "layout.tsx"}:
                return True

    # Fallback: top-level executable script with "cli" or "server" in name.
    if len(parts) == 1 and any(k in name for k in ("cli", "server", "main")):
        return True

    return False


def _is_cli_like(rel_path: str, card: Dict[str, Any], lang: str) -> bool:
    """
    Heuristic: is this file likely to be a CLI tool?
    """
    p = Path(rel_path)
    name = p.name.lower()
    parts = [part.lower() for part in p.parts]

    # Name hints
    if any(k in name for k in ("cli", "tool", "script")):
        return True

    # Directory hints
    if any(k in parts for k in ("bin", "scripts", "tools")):
        return True

    # Card hints (if analyzer already picked up CLI args)
    cli_args = card.get("cli_args")
    if isinstance(cli_args, list) and cli_args:
        return True

    # Python "if __name__ == '__main__'" — only if we have content in card
    # (We keep this simple to avoid reading heavy files here.)
    if lang == "python":
        summary = card.get("summary") or {}
        if isinstance(summary, dict):
            txt = (summary.get("heuristic") or summary.get("ai_text") or "").lower()
            if "__main__" in txt:
                return True

    return False


def _is_probably_binary(path: Path) -> bool:
    """
    Quick extension-based check to avoid reading binary/content-heavy files.
    """
    return path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".zip",
        ".jar",
        ".exe",
        ".dll",
    }


def _is_skipped_path(path: Path) -> bool:
    """
    Skip paths based on directory, basename, or suffix.
    """
    # Skip by directories
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    # Skip by basename/suffix
    name = path.name.lower()
    if name in SKIP_BASENAMES:
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def _first_lines(s: str, n_lines: int = 10) -> str:
    """
    Fallback heuristic: first N lines, capped to SUMMARY_MAX_CHARS.
    """
    lines = s.splitlines()[:n_lines]
    out = "\n".join(lines).strip()
    if len(out) > SUMMARY_MAX_CHARS:
        out = out[:SUMMARY_MAX_CHARS].rstrip() + "…"
    return out


def _tidy_summary(text: str, max_len: int = SUMMARY_MAX_CHARS) -> str:
    """
    Collapse whitespace/newlines; hard-cap length; append ellipsis if trimmed.
    """
    s = " ".join(text.split())
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def _safe_json_load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_cards_index(raw: Any) -> Dict[str, Any]:
    """
    Accept several shapes and normalize to { rel_path: node_dict }.

    Expected fields (when present) on each node:
      - summary: { heuristic, ai_text, ai_json, ai_model, ai_ts, ai_sha }
      - staleness: { changed, needs_ai_refresh }
      - language / lang
      - routes / cli_args / env_vars / public_api (optional)
    """
    if isinstance(raw, dict):
        # shape: { "cards": { "path": {...}, ... } }
        if "cards" in raw and isinstance(raw["cards"], dict):
            return raw["cards"]
        # Flat list shape: { "files": [ { "path": "..." }, ... ] }
        if "files" in raw and isinstance(raw["files"], list):
            out: Dict[str, Any] = {}
            for node in raw["files"]:
                if isinstance(node, dict) and "path" in node:
                    out[str(node["path"])] = node
            return out
        # Already keyed by path
        return raw
    return {}


def _read_cards_index(project_root: Path) -> Dict[str, Any]:
    """
    Read the canonical cards index, preferring .aidev/cards/index.json.

    Legacy fallbacks are kept for compatibility:
      - .aidev/cards/index.json   # canonical
      - .aidev/cards.json         # legacy
      - .cards.json               # very old legacy

    NOTE: we deliberately do NOT read .aidev/index.json here; that file is a
    separate, richer structure used elsewhere.
    """
    candidates = [
        project_root / ".aidev" / "cards" / "index.json",  # canonical
        project_root / ".aidev" / "cards.json",            # legacy
        project_root / ".cards.json",                      # very old legacy
    ]
    for p in candidates:
        if p.exists():
            raw = _safe_json_load(p)
            if raw is None:
                continue
            idx = _normalize_cards_index(raw)
            if isinstance(idx, dict):
                return idx
    return {}


def _project_map_path(project_root: Path) -> Path:
    return project_root / PROJECT_MAP_REL


def _load_project_map(project_root: Path) -> Optional[Dict[str, Any]]:
    p = _project_map_path(project_root)
    if not p.exists():
        return None
    return _safe_json_load(p)


def _serialize(obj: Dict[str, Any]) -> str:
    """
    Stable formatting for equality checks; indent for readability.
    """
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)


def _save_if_changed(project_root: Path, data: Dict[str, Any], *, force: bool = False) -> Path:
    """
    Write .aidev/project_map.json if content changed, unless force=True.

    - force=False (default): read existing content and only replace on diff.
    - force=True: unconditionally rewrite the file (useful for debugging or
      when external tooling cares about mtime even if content is identical).
    """
    p = _project_map_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_text = _serialize(data)

    if not force and p.exists():
        try:
            old_text = p.read_text(encoding="utf-8")
        except Exception:
            old_text = ""
        else:
            if new_text == old_text:
                return p

    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(p)
    return p


def _ai_summary_from_card(card: Dict[str, Any]) -> str:
    """
    Prefer the AI summary text from card['summary']['ai_text'], falling back to
    card['summary']['heuristic']. Returns "" if neither is available.
    """
    summary = card.get("summary")
    if isinstance(summary, dict):
        ai_text = summary.get("ai_text")
        if isinstance(ai_text, str) and ai_text.strip():
            return ai_text.strip()
        heur = summary.get("heuristic")
        if isinstance(heur, str) and heur.strip():
            return heur.strip()
    return ""


def _heuristic_summary_for(path: Path, text: str, lang: str) -> str:
    """
    Tiny heuristic summary used only when no AI/heuristic summary is available
    in the card itself:
      - For Python: try module docstring first.
      - Otherwise: first lines of the file.
    """
    if lang == "python":
        try:
            mod = ast.parse(text)
            doc = ast.get_docstring(mod) or ""
            if doc:
                trimmed = doc.strip()
                if len(trimmed) > SUMMARY_MAX_CHARS:
                    return trimmed[:SUMMARY_MAX_CHARS].rstrip() + "…"
                return trimmed
        except Exception:
            # Fall back to first lines if parsing or docstring extraction fails
            pass
    return _first_lines(text, 10)


def _collect_known_paths(cards: Dict[str, Any]) -> List[str]:
    """
    Normalize card keys into stable, POSIX-style relative paths.
    """
    return sorted({str(Path(p).as_posix()).lstrip("./") for p in cards.keys()})


def _cap_list(x, k: int = 8):
    """
    Take up to k unique items from a list, preserving order.
    Used to keep hint lists tiny in the project map.
    """
    if not isinstance(x, list):
        return None
    out: List[Any] = []
    for v in x:
        if v in out:
            continue
        out.append(v)
        if len(out) >= k:
            break
    return out


def _rank_path(path: str) -> int:
    """
    Stable, meaningful ordering: nudge key entrypoints up, then aidev/*,
    then everything else. Lower number = earlier.
    """
    tops = [
        "aidev_cli.py",
        "aidev/core.py",
        "aidev/server.py",
        "aidev/orchestrator.py",
        "aidev/ui/app.js",
        "aidev/api/",
        "aidev/routes/",
    ]
    for i, k in enumerate(tops):
        if path.startswith(k) or path.endswith(k):
            return i
    # Prefer project code (aidev/*) before other top-level files
    return len(tops) + (0 if path.startswith("aidev/") else 1000)


def _top_segment(rel_path: str) -> str:
    """
    First path segment for aggregation, e.g.:

      "aidev/repo_map.py" -> "aidev"
      "src/app.tsx"       -> "src"
      "server.py"         -> "server.py"
    """
    parts = Path(rel_path).parts
    if not parts:
        return rel_path
    return parts[0]


# ----------------------------
# Core build
# ----------------------------

def build_project_map(
    project_root: Path, force: bool = False, max_files: Optional[int] = None, include_generated: bool = False
) -> Dict[str, Any]:
    """
    Build or update .aidev/project_map.json (LLM-facing).

    - Uses .aidev/cards/index.json as the source of truth.
    - Prefers card['summary']['ai_text'] as the summary text, falling back to:
        card['summary']['heuristic'] -> tiny on-the-fly heuristic.
    - Emits ONLY a compact, LLM-facing structure with repo-level aggregates.

    Args:
        max_files: Optional cap on number of file entries emitted.
        include_generated: If True, allow generated/internal paths (e.g. .aidev/*) into the map.

    Returned dict:
        {
            "version": 1,
            "generated_at": ISO-UTC,
            "root": "<absolute project_root>",
            "total_files": <int>,
            "language_kinds": { "<language>": <count>, ... },
            "by_ext": { "<.ext>": <count>, ... },
            "by_top": { "<top>": <count>, ... },
            "files": [
                {
                    "path": "...",
                    "language": "...",
                    "kind": "code" | "test" | "config" | "doc" | "ui" | "asset" | "other",
                    "summary": "...",
                    "routes": [...?],
                    "cli_args": [...?],
                    "env_vars": [...?],
                    "public_api": [...?],
                    "changed": true,  # optional
                },
                ...
            ]
        }

    The returned dict is the in-memory representation; the on-disk JSON is
    written via _save_if_changed(project_root, result, force=force).
    """
    project_root = project_root.resolve()
    cards = _read_cards_index(project_root)
    known_paths = _collect_known_paths(cards)

    # Filter out generated/internal artifacts early unless explicitly requested.
    # This keeps downstream prompt payloads clean and avoids accidental inclusion
    # of .aidev/* content.
    if not include_generated:
        known_paths = [
            rel for rel in known_paths if not _is_skipped_path(project_root / rel)
        ]

    total_known_files = len(known_paths)
    effective_max_files = PROJECT_MAP_MAX_FILES if max_files is None else max_files

    files_list: List[Dict[str, Any]] = []

    # Aggregates
    language_counts: Dict[str, int] = {}
    ext_counts: Dict[str, int] = {}
    top_counts: Dict[str, int] = {}

    for rel in known_paths:
        card = cards.get(rel)
        # Graceful fallback for older indexes keyed as "./path"
        if card is None and ("./" + rel) in cards:
            card = cards.get("./" + rel)
        if not isinstance(card, dict):
            card = {}
        abs_path = project_root / rel

        # Skip internal / binary / noisy files
        if _is_probably_binary(abs_path):
            continue

        # Prefer language from card; fall back to extension
        lang = (
            str(card.get("language") or "").strip()
            or str(card.get("lang") or "").strip()
            or _lang_for_path(abs_path)
        )

        # Prefer the LLM-written card summary; fall back to a tiny heuristic
        summary = (_ai_summary_from_card(card) or "").strip()
        if not summary:
            text = _read_text_safe(abs_path, max_bytes=120_000)
            summary = _heuristic_summary_for(abs_path, text, lang)

        summary = _tidy_summary(summary, SUMMARY_MAX_CHARS)

        # Prefer kind from card if it's one of our known categories,
        # otherwise fall back to the path/extension heuristic.
        card_kind = str(card.get("kind") or "").strip().lower()
        if card_kind in {"code", "test", "config", "doc", "ui", "asset", "other"}:
            kind = card_kind
        else:
            kind = _kind_for_path(rel, lang)

        item: Dict[str, Any] = {
            "path": rel,
            "language": lang,
            "kind": kind,
            "summary": summary,
        }

        # Optional "changed" hint from v2 staleness; tiny and useful downstream.
        st = card.get("staleness") or {}
        if isinstance(st, dict) and st.get("changed"):
            item["changed"] = True

        # Derived tags (tiny, token-lean) to help planners / prompts.
        tags: List[str] = []

        if kind == "test":
            tags.append("test")
        if kind == "config":
            tags.append("config")
        if _is_entrypoint(rel, lang):
            tags.append("entrypoint")
        if _is_cli_like(rel, card, lang):
            tags.append("cli")

        # Only attach if non-empty to keep JSON lean.
        if tags:
            item["tags"] = tags

        # Optional small hints from cards (hard-capped and only if present; drop empties)
        if isinstance(card, dict):
            for k in HINT_KEYS:
                v = card.get(k)
                v = _cap_list(v, 8) if isinstance(v, list) else v
                if v:
                    item[k] = v

        files_list.append(item)

        # Aggregates
        language_counts[lang] = language_counts.get(lang, 0) + 1
        ext = abs_path.suffix.lower()
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        top = _top_segment(rel)
        top_counts[top] = top_counts.get(top, 0) + 1

    # Stable, meaningful ordering
    files_list.sort(key=lambda f: (_rank_path(f["path"]), f["path"]))

    truncated = False
    if isinstance(effective_max_files, int) and effective_max_files >= 0:
        if len(files_list) > effective_max_files:
            files_list = files_list[:effective_max_files]
            truncated = True

    result = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(project_root),
        "total_files": len(files_list),
        "total_known_files": total_known_files,
        "max_files": effective_max_files,
        "truncated": truncated,
        "language_kinds": dict(sorted(language_counts.items())),
        "by_ext": dict(sorted(ext_counts.items())),
        "by_top": dict(sorted(top_counts.items())),
        "files": files_list,
    }

    # Write only if changed (or always, if force=True)
    _save_if_changed(project_root, result, force=force)
    return result


# ----------------------------
# Repo-level heuristics
# ----------------------------

def summarize_repo_from_map(project_root: Path) -> Dict[str, Any]:
    """
    Rebuild .aidev/project_map.json from current cards, then compute repo-level hints.

    build_project_map() is idempotent and only rewrites the file if content
    actually changed (unless force=True is used explicitly elsewhere).
    """
    project_root = project_root.resolve()
    project_map = build_project_map(project_root)

    files = project_map.get("files") or []
    languages = sorted({(f.get("language") or "text") for f in files})

    paths = [str(f.get("path") or "") for f in files]
    lowered = [p.lower() for p in paths]

    frameworks: set[str] = set()
    tools: set[str] = set()

    # Framework / stack hints
    if any(p.endswith("pubspec.yaml") for p in lowered) or any(p.endswith(".dart") for p in lowered):
        frameworks.add("flutter")
    if any(p.endswith("package.json") for p in lowered):
        frameworks.add("node/react/nextjs")
    if any(p.endswith("composer.json") for p in lowered):
        frameworks.add("php (composer)")
    if any(p.endswith("pyproject.toml") or p.endswith("requirements.txt") for p in lowered):
        frameworks.add("python")

    # Tooling hints
    if any("tsconfig" in p for p in lowered):
        tools.add("typescript")
    if any("ruff.toml" in p for p in lowered):
        tools.add("ruff")
    if any(".eslintrc" in p or "eslint.config" in p for p in lowered):
        tools.add("eslint")
    if any(".prettierrc" in p or "prettier.config" in p for p in lowered):
        tools.add("prettier")

    return {
        "languages": languages,
        "frameworks": sorted(frameworks),
        "tools": sorted(tools),
        "file_count": len(files),
    }


# ----------------------------
# CLI
# ----------------------------

def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build or update .aidev/project_map.json (LLM-lean)"
    )
    ap.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Repo root",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force write even if content unchanged (useful for debugging)",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap number of file entries emitted into the project map",
    )
    ap.add_argument(
        "--include-generated",
        action="store_true",
        help="Include generated/internal paths (e.g. .aidev/*) in the project map",
    )
    ap.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print the resulting JSON path and file count",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Print the resulting JSON to stdout",
    )
    args = ap.parse_args(argv)

    root = Path(args.project_root).resolve()
    result = build_project_map(
        root,
        force=args.force,
        max_files=args.max_files,
        include_generated=args.include_generated,
    )

    if args.do_print:
        p = _project_map_path(root)
        _print(
            json.dumps(
                {"path": str(p), "files": len(result.get("files", [])), "truncated": bool(result.get("truncated"))},
                indent=2,
            )
        )
    if args.show:
        _print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())