# aidev/cards.py
from __future__ import annotations

import concurrent.futures
import fnmatch
import hashlib
import json
import logging as _logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import (
    ai_summary_schema,
    SchemaNotFoundError,
    SchemaFormatError,
    SchemaValidationUnavailable,
)


# -------- Logger (same shim style as llm_client) --------
try:
    from . import logger  # type: ignore
except Exception:
    _log = _logging.getLogger("aidev.cards")

    class _Shim:
        def info(self, msg, ctx=None):  # type: ignore[override]
            _log.info(f"{msg} | {ctx or {}}")

        def warn(self, msg, ctx=None, exc=None):  # type: ignore[override]
            _log.warning(f"{msg} | {ctx or {}}")

        def warning(self, msg, ctx=None, exc=None):  # type: ignore[override]
            _log.warning(f"{msg} | {ctx or {}}")

        def error(self, msg, ctx=None, exc=None):  # type: ignore[override]
            _log.error(f"{msg} | {ctx or {}}")

        def exception(self, msg, ctx=None):  # type: ignore[override]
            _log.exception(f"{msg} | {ctx or {}}")

    logger = _Shim()

# Optional import of aidev.config (preferred source for REFRESH_CARDS_BETWEEN_RECS)
try:
    from . import config as _cfg  # type: ignore
except Exception:
    class _CfgShim:
        REFRESH_CARDS_BETWEEN_RECS = True
        SESSION_ID = None

    _cfg = _CfgShim()

# -------- Internal denylist globs (never summarize/index these) --------
_INTERNAL_DENY_GLOBS: List[str] = [
    # internal/generated
    "**/.aidev/**",
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/*.card.json",
    # housekeeping / secrets (block root and nested variants)
    "*.log",
    "**/*.log",
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".prettierrc",
    ".prettierrc.*",
    ".npmrc",
    ".dockerignore",
    ".aidev.json",
    ".DS_Store",
    "Thumbs.db",
    "**/*.lock",
    "**/package-lock.json",
    "**/pnpm-lock.yaml",
    "**/yarn.lock",
    "**/.ruff_cache/**",
    "**/.mypy_cache/**",
]


def _to_posix(path: str) -> str:
    return path.replace("\\", "/")


def _matches_any(path_posix: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path_posix, pat) for pat in patterns)


# Optional embeddings (safe no-op if missing)
try:
    from .llm_client import embed_texts as _llm_embed_texts  # type: ignore
except Exception:
    _llm_embed_texts = None  # type: ignore

# Optional JSON Schema validation
try:
    from jsonschema import Draft202012Validator as _Validator  # type: ignore

    _HAVE_JSONSCHEMA = True
except Exception:
    _Validator = None  # type: ignore
    _HAVE_JSONSCHEMA = False

__all__ = [
    "Card",
    "KnowledgeBase",
    # forwarders
    "summarize_changed",
]

# ----------------- Data types -----------------


@dataclass
class Card:
    id: str
    title: str
    prompt: str
    includes: List[str] = field(default_factory=list)  # glob rel paths
    excludes: List[str] = field(default_factory=list)
    budget: int = 4096
    top_k: int = 20
    acceptance_criteria: List[str] = field(default_factory=list)


# ----------------- Small helpers -----------------

GENERATOR_VERSION = os.getenv("AIDEV_GENERATOR_VERSION", "aidev.cards.v2")
SCHEMA_VERSION = 2


def _now_iso() -> str:
    return (
        datetime.utcnow()
        .replace(tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalize_for_hash(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on", "y"}


# ----------------- Extraction helpers -----------------

_TODO_RE = re.compile(
    r"(?im)^\s*(//+|#|<!--|/\*+)?\s*(todo|fixme|hack)[:\s-]+(.+?)(?:-->|$|\*/)",
    re.IGNORECASE,
)

# Python, JS/TS signatures
_PY_DEF_RE = re.compile(
    r"(?m)^\s*async\s+def\s+([A-Za-z_][A-Za-z0-9_]*)|^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_PY_CLASS_RE = re.compile(r"(?m)^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")
_JS_EXPORT_RE = re.compile(
    r"export\s+(?:default\s+)?(class|function|const|let|var)\s+([A-Za-z0-9_]+)"
)
_JS_NAMED_EXPORTS_RE = re.compile(r"export\s*\{\s*([^}]+)\s*\}")
_TS_TYPE_EXPORT_RE = re.compile(r"export\s+type\s+([A-Za-z0-9_]+)")

# Web/API routes: Express/FastAPI/Flask
_EXPRESS_ROUTES = re.compile(
    r"(?m)\b(?:app|router)\.(get|post|put|delete|patch|options|head)\s*\(\s*[\'\"]([^\'\"]+)[\'\"]"
)
_FASTAPI_ROUTES = re.compile(
    r"(?m)@(?:app|router)\.(get|post|put|delete|patch|options|head)\s*\(\s*[\'\"]([^\'\"]+)[\'\"]"
)
_FLASK_ROUTES = re.compile(
    r"(?m)@(?:app|bp)\.route\s*\(\s*[\'\"]([^\'\"]+)[\'\"]"
)

# CLI args (argparse)
_ARGPARSE_FLAGS = re.compile(r"add_argument\s*\([^)]*\)")
_FLAG_TOKENS = re.compile(r"[\'\"](-{1,2}[A-Za-z0-9][A-Za-z0-9_\-]*)[\'\"]")

# Env vars across ecosystems
_ENV_VARS = re.compile(
    r"(?:process\.env\.([A-Za-z0-9_]+)|os\.getenv\(\s*[\'\"]([A-Za-z0-9_]+)[\'\"]\s*\)|"
    r"os\.environ\[\s*[\'\"]([A-Za-z0-9_]+)[\'\"]\s*\]|Deno\.env\.get\(\s*[\'\"]([A-Za-z0-9_]+)[\'\"]\s*\)|"
    r"import\.meta\.env\.([A-Za-z0-9_]+))"
)

# ---------- Language detection ----------


def _language_for_path(rel: str) -> str:
    low = rel.lower()
    for ext, lang in (
        (".py", "python"),
        (".ipynb", "python"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".js", "javascript"),
        (".jsx", "javascript"),
        (".rs", "rust"),
        (".go", "go"),
        (".java", "java"),
        (".kt", "kotlin"),
        (".kts", "kotlin"),
        (".swift", "swift"),
        (".php", "php"),
        (".c", "c"),
        (".h", "c"),
        (".cpp", "cpp"),
        (".cc", "cpp"),
        (".hpp", "cpp"),
        (".rb", "ruby"),
        (".cs", "csharp"),
        (".css", "css"),
        (".scss", "scss"),
        (".less", "less"),
        (".html", "html"),
        (".xml", "xml"),
        (".json", "json"),
        (".yaml", "yaml"),
        (".yml", "yaml"),
        (".md", "markdown"),
        (".mdx", "markdown"),
        (".sql", "sql"),
        (".toml", "toml"),
        (".ini", "ini"),
        (".sh", "bash"),
        (".zsh", "zsh"),
        (".ps1", "powershell"),
    ):
        if low.endswith(ext):
            return lang
    return "other"


def _first_heading_or_sentence(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 12:
            return s[:200]
    return ""


def _first_docblock_or_comment(text: str, rel_path: str) -> str:
    low = rel_path.lower()
    if low.endswith(".py"):
        m = re.search(r'^\s*"""(.*?)"""', text, re.S) or re.search(
            r"^\s*'''(.*?)'''", text, re.S
        )
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:240]
    m = re.search(r"/\*\s*(.*?)\s*\*/", text, re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:240]
    m = re.search(r"^\s*//\s*(.+)$", text, re.M)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:240]
    return ""


def _extract_imports(text: str) -> List[str]:
    res: List[str] = []
    # ES/TS imports & requires
    for m in re.finditer(
        r'(?:from\s+["\']([^"\']+)["\']|import\s+([A-Za-z0-9_./\-]+)|require\(\s*["\']([^"\']+)["\']\s*\))',
        text,
    ):
        s = m.group(1) or m.group(2) or m.group(3)
        if s:
            res.append(s.strip())
    # Python: from pkg.subpkg import name
    for m in re.finditer(r"\bfrom\s+([A-Za-z0-9_.]+)\s+import\b", text):
        res.append(m.group(1).strip())
    # using Foo.Bar;
    for m in re.finditer(r"(?m)^\s*using\s+([A-Za-z0-9_.]+)\s*;", text):
        res.append(m.group(1).strip())
    # C/C++ includes
    for m in re.finditer(r'(?m)^\s*#\s*include\s*[<"]([^>"]+)[>"]', text):
        res.append(m.group(1).strip())
    # PHP include/require
    for m in re.finditer(
        r'(?:require|include|require_once|include_once)\(\s*["\']([^"\']+)["\']\s*\)',
        text,
    ):
        res.append(m.group(1).strip())
    # Keep it small
    out: List[str] = []
    seen: set[str] = set()
    for s in res:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= 64:
            break
    return out


def _extract_exports_and_symbols(text: str, rel: str) -> Tuple[List[str], List[str]]:
    exports: List[str] = []
    symbols: List[str] = []
    # JS/TS exports
    for _kind, name in _JS_EXPORT_RE.findall(text):
        exports.append(name)
        symbols.append(name)
    for m in _JS_NAMED_EXPORTS_RE.finditer(text):
        for name in m.group(1).split(","):
            nm = name.strip()
            if not nm:
                continue
            left = nm.split(" as ")
            if len(left) == 2:
                symbols.append(left[0].strip())
                exports.append(left[1].strip())
            else:
                symbols.append(nm)
                exports.append(nm)
    for m in _TS_TYPE_EXPORT_RE.finditer(text):
        nm = m.group(1)
        exports.append(nm)
        symbols.append(nm)
    # Python defs/classes
    for a, b in _PY_DEF_RE.findall(text):
        nm = a or b
        if nm:
            symbols.append(nm)
    for m in _PY_CLASS_RE.finditer(text):
        symbols.append(m.group(1))
    # Fallback: common const (JS)
    for m in re.finditer(
        r"(?m)^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", text
    ):
        symbols.append(m.group(1))

    # Normalize/limit
    def _clean(xs: List[str], k: int) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for x in xs:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
            if len(out) >= k:
                break
        return out

    return _clean(exports, 64), _clean(symbols, 96)


def _extract_todos(text: str) -> List[str]:
    out: List[str] = []
    for m in _TODO_RE.finditer(text):
        msg = (m.group(3) or "").strip()
        if msg:
            out.append(msg[:180])
        if len(out) >= 64:
            break
    # dedupe
    seen: set[str] = set()
    dedup: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


def _extract_routes(text: str) -> List[str]:
    routes: List[str] = []
    for m in _EXPRESS_ROUTES.finditer(text):
        routes.append(f"{m.group(1).upper()} {m.group(2)}")
    for m in _FASTAPI_ROUTES.finditer(text):
        routes.append(f"{m.group(1).upper()} {m.group(2)}")
    for m in _FLASK_ROUTES.finditer(text):
        routes.append(f"ROUTE {m.group(1)}")
    # dedupe
    seen: set[str] = set()
    out: List[str] = []
    for r in routes:
        if r not in seen:
            seen.add(r)
            out.append(r)
        if len(out) >= 64:
            break
    return out


def _extract_cli_args(text: str) -> List[str]:
    out: List[str] = []
    for block in _ARGPARSE_FLAGS.findall(text):
        out.extend(_FLAG_TOKENS.findall(block))
    # dedupe
    seen: set[str] = set()
    dedup: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
        if len(dedup) >= 64:
            break
    return dedup


def _extract_env_vars(text: str) -> List[str]:
    found: List[str] = []
    for m in _ENV_VARS.finditer(text):
        for v in m.groups():
            if v:
                found.append(v)
    seen: set[str] = set()
    out: List[str] = []
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= 64:
            break
    return out


def _heuristic_summary(
    text: str, rel_path: str, imports: List[str], exports: List[str], symbols: List[str]
) -> str:
    s = ""
    if rel_path.lower().endswith((".md", ".mdx", ".txt", ".rst", ".adoc")):
        s = _first_heading_or_sentence(text)
    if not s:
        s = _first_docblock_or_comment(text, rel_path)
    if not s:
        base = os.path.basename(rel_path)
        exp = ", ".join(exports[:3]) + ("…" if len(exports) > 3 else "")
        sym = ", ".join(symbols[:3]) + ("…" if len(symbols) > 3 else "")
        imp = ", ".join([i.rsplit("/", 1)[-1] for i in imports[:3]]) + (
            "…" if len(imports) > 3 else ""
        )
        s = (
            f"{base} — exports({exp or 'none'}); "
            f"symbols({sym or 'none'}); imports({imp or 'none'})."
        )
    return s[:280]


# --- TS/JS path resolution + CODEOWNERS + git last-change (optional, best-effort) ---


def _load_tsconfig(root: Path) -> Dict[str, Any]:
    for name in ("tsconfig.json", "clients/web/tsconfig.json"):
        p = root / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                return {}
    return {}


def _resolve_import_specifier(
    spec: str, rel_from: Path, root: Path, tsconf: Dict[str, Any]
) -> Optional[str]:
    spec = (spec or "").strip()
    if not spec or spec.startswith(("http://", "https://")):
        return None

    # Relative
    if spec.startswith("."):
        cand = (rel_from.parent / spec).resolve()
        for ext in ("", ".ts", ".tsx", ".js", ".jsx", ".py"):
            p = Path(str(cand) + ext)
            if p.exists() and root in p.parents:
                return p.relative_to(root).as_posix()
        idx = cand / "index.ts"
        if idx.exists() and root in idx.parents:
            return idx.relative_to(root).as_posix()
        return None

    # tsconfig baseUrl/paths
    try:
        co = tsconf.get("compilerOptions", {})
        base = co.get("baseUrl", "")
        base_dir = (root / base).resolve() if base else root
        paths = co.get("paths") or {}
        for patt, arr in paths.items():
            star = "*" in patt
            if (star and spec.startswith(patt.split("*", 1)[0])) or (not star and patt == spec):
                for target in arr or []:
                    repl = target.replace("*", spec.split(patt.replace("*", ""), 1)[-1]) if star else target
                    cand = (base_dir / repl).resolve()
                    for ext in ("", ".ts", ".tsx", ".js", ".jsx"):
                        p = Path(str(cand) + ext)
                        if p.exists() and root in p.parents:
                            return p.relative_to(root).as_posix()
                    idx = cand / "index.ts"
                    if idx.exists() and root in idx.parents:
                        return idx.relative_to(root).as_posix()
        cand = (base_dir / spec).resolve()
        for ext in ("", ".ts", ".tsx", ".js", ".jsx"):
            p = Path(str(cand) + ext)
            if p.exists() and root in p.parents:
                return p.relative_to(root).as_posix()
        idx = cand / "index.ts"
        if idx.exists() and root in idx.parents:
            return idx.relative_to(root).as_posix()
    except Exception:
        return None
    return None


def _parse_codeowners(root: Path) -> List[Tuple[str, List[str]]]:
    for loc in ("CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"):
        p = root / loc
        if p.exists():
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                out: List[Tuple[str, List[str]]] = []
                for ln in lines:
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if not parts:
                        continue
                    pat, owners = parts[0], [x for x in parts[1:] if x.startswith("@")]
                    if owners:
                        out.append((pat, owners))
                return out
            except Exception:
                return []
    return []


def _owners_for_path(rel: str, rules: List[Tuple[str, List[str]]]) -> List[str]:
    best: Tuple[int, List[str]] | None = None
    for pat, owners in rules:
        if fnmatch.fnmatch(rel, pat):
            score = len(pat)
            if best is None or score > best[0]:
                best = (score, owners)
    return best[1] if best else []


def _git_last_change(root: Path, rel: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%ct|%an", "--", rel],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            ts_s, author = (cp.stdout.strip().split("|", 1) + [""])[:2]
            return int(ts_s), author
    except Exception:
        pass
    return None, None


# ----------------- Knowledge base -----------------


@dataclass
class KnowledgeBase:
    """
    Lightweight file index with glob filtering + card cache integration.

    `structure` is mapping of rel_path -> kind (from discover_structure()).
    """

    root: Path
    structure: Dict[str, str]

    # ---------- Schema / validator cache ----------
    _card_schema_validator: Any = field(default=None, init=False, repr=False)
    _ai_schema_validator: Any = field(default=None, init=False, repr=False)

    
    def _load_validator(self, name: str) -> Any:
        """
        Lazily construct and return a jsonschema validator for the given schema.

        Returns:
            A validator instance, or None if validation is unavailable
            (e.g., jsonschema not installed, schema missing/invalid).
        """
        if not _HAVE_JSONSCHEMA:
            return None

        schema: Dict[str, Any]

        if name == "ai_summary":
            # Prefer the central helper for the AI summary schema.
            try:
                schema = ai_summary_schema()
            except (SchemaNotFoundError, SchemaFormatError, SchemaValidationUnavailable) as exc:
                logger.warning(
                    "AI schema validation disabled",
                    ctx={"name": name, "err": str(exc)[:200]},
                )
                return None
            except Exception as exc:
                logger.warning(
                    "Unexpected error loading AI summary schema",
                    ctx={"name": name, "err": str(exc)[:200]},
                )
                return None
        else:
            # Existing behavior for repo-local JSON schema files (e.g. cards.schema.json)
            schema_path = self.root / f"{name}.schema.json"
            if not schema_path.exists():
                alt = self.root / ".aidev" / "schemas" / f"{name}.schema.json"
                schema_path = alt if alt.exists() else schema_path
            if not schema_path.exists():
                return None
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                logger.warning(
                    "Schema load failed",
                    ctx={"name": name, "err": str(exc)[:200]},
                )
                return None

        try:
            return _Validator(schema)
        except Exception as exc:
            logger.warning(
                "Schema validator construction failed",
                ctx={"name": name, "err": str(exc)[:200]},
            )
            return None

    def _get_card_validator(self) -> Any:
        if self._card_schema_validator is None:
            self._card_schema_validator = self._load_validator("cards")
        return self._card_schema_validator
    def _get_ai_validator(self) -> Any:
        if self._ai_schema_validator is None:
            self._ai_schema_validator = self._load_validator("ai_summary")
        return self._ai_schema_validator

    # ---------- Basic file selection ----------
    def select(self, includes: List[str], excludes: List[str]) -> List[str]:
        """
        Return a sorted list of relative paths matching includes and not matching excludes,
        with internal/generated files always filtered out.
        """
        all_files = list(self.structure.keys())

        # Start from include set (or everything if unspecified)
        if includes:
            pool = [
                rel
                for rel in all_files
                if any(fnmatch.fnmatch(_to_posix(rel), g) for g in includes)
            ]
        else:
            pool = list(all_files)

        # Always exclude internal/generated files
        pool = [
            rel
            for rel in pool
            if not _matches_any(_to_posix(rel), _INTERNAL_DENY_GLOBS)
        ]

        # Apply caller-provided excludes too
        if excludes:
            pool = [
                rel
                for rel in pool
                if not any(fnmatch.fnmatch(_to_posix(rel), g) for g in excludes)
            ]

        return sorted(pool)
    # ---------- Card/index paths ----------
    @property
    def _cards_dir(self) -> Path:
        p = self.root / ".aidev" / "cards"
        p.mkdir(parents=True, exist_ok=True)
        return p
    @property
    def _cards_index_path(self) -> Path:
        return self._cards_dir / "index.json"
    def _card_json_path(self, rel_path: str) -> Path:
        safe_rel = Path(rel_path).as_posix()
        p = self._cards_dir / f"{safe_rel}.card.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    @property
    def _structure_cache_path(self) -> Path:
        return self.root / ".aidev" / "cache" / "file_index.json"
    @property
    def _graph_index_path(self) -> Path:
        return self.root / ".aidev" / "index.json"
    # ---------- Summary helpers (new structured) ----------
    @staticmethod
    def _ensure_summary_obj(meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize meta['summary'] to a dict with all expected keys.

        - Accepts legacy string summaries and wraps them as {'heuristic': ...}.
        - Does NOT overwrite existing ai_* fields (only fills missing ones).
        - Optionally imports legacy flat ai_summary* fields.
        """
        s = meta.get("summary")

        if isinstance(s, str):
            s = {"heuristic": s or ""}

        if not isinstance(s, dict):
            s = {}

        # Fill defaults without clobbering any existing ai_* values
        s.setdefault("heuristic", "")
        s.setdefault("ai_text", s.get("ai_text"))
        s.setdefault("ai_json", s.get("ai_json"))
        s.setdefault("ai_model", s.get("ai_model"))
        s.setdefault("ai_ts", s.get("ai_ts"))
        s.setdefault("ai_sha", s.get("ai_sha"))

        # Optional carry-in from legacy flat fields (for old indexes/cards)
        if (
            meta.get("ai_summary")
            or meta.get("ai_summary_model")
            or meta.get("ai_summary_ts")
            or meta.get("ai_summary_sha")
        ):
            if not s.get("ai_text") and isinstance(meta.get("ai_summary"), str):
                s["ai_text"] = meta.get("ai_summary")
            if not s.get("ai_model") and isinstance(meta.get("ai_summary_model"), str):
                s["ai_model"] = meta.get("ai_summary_model")
            if not s.get("ai_ts") and isinstance(meta.get("ai_summary_ts"), str):
                s["ai_ts"] = meta.get("ai_summary_ts")
            if not s.get("ai_sha") and isinstance(meta.get("ai_summary_sha"), str):
                s["ai_sha"] = meta.get("ai_summary_sha")

        meta["summary"] = s
        return s

    def _effective_summary_text(self, meta: Any) -> str:
        """
        Extract a readable summary string from a card meta dict or related structure.

        Precedence is:
            - ai_summary / summary.ai_text / summary.ai_summary
            - summary.heuristic
            - common nested content/text fields
        with a generic JSON fallback for structured objects.
        """
        def _coerce(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            if isinstance(x, dict):
                # top-level checks
                for key in ("ai_text", "ai_summary", "text", "heuristic"):
                    v = x.get(key)
                    if isinstance(v, str):
                        return v
                # dive into common nested locations (incl. summary dict)
                for key in ("summary", "text", "value", "content"):
                    v = x.get(key)
                    if isinstance(v, dict):
                        y = _coerce(v)
                        if y:
                            return y
                # fallback to JSON for structured blobs
                return json.dumps(x, ensure_ascii=False)
            if isinstance(x, (list, tuple)):
                for item in x:
                    y = _coerce(item)
                    if y:
                        return y
            return str(x)

        s = " ".join((_coerce(meta) or "").split())
        return s

    @staticmethod
    def _set_ai_summary(
        meta: Dict[str, Any],
        *,
        ai_text: Optional[str],
        ai_json: Optional[dict],
        model: Optional[str],
        file_sha: str,
    ) -> Dict[str, Any]:
        """
        Attach AI summary fields to meta['summary'] and return the updated meta.
        """
        s_before = KnowledgeBase._ensure_summary_obj(meta)

        if ai_text is not None:
            s_before["ai_text"] = (ai_text or "").strip() or None
        if ai_json is not None:
            s_before["ai_json"] = ai_json

        s_before["ai_model"] = model
        s_before["ai_ts"] = _now_iso()
        s_before["ai_sha"] = file_sha

        meta["summary"] = s_before

        return meta

    @staticmethod
    def _apply_structured_summary(
        meta: Dict[str, Any],
        ai_json: Dict[str, Any],
    ) -> None:
        """
        Project structured AI summary (what/why/how/public_api/key_deps/risks/tests/notes)
        into richer card fields: summaries, contracts, metrics, key_deps.

        This does NOT touch summary.ai_* fields; those are handled by _set_ai_summary.
        """
        def _as_list(v: Any) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if x]
            return [str(v)] if v else []

        # Ensure sub-objects exist
        summaries = meta.get("summaries")
        if not isinstance(summaries, dict):
            summaries = {}
        contracts = meta.get("contracts")
        if not isinstance(contracts, dict):
            contracts = {}
        metrics = meta.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        # --- Summaries: short + long ---
        what = _as_list(ai_json.get("what"))
        why = _as_list(ai_json.get("why"))
        how = _as_list(ai_json.get("how"))

        if what and not summaries.get("summary_short"):
            summaries["summary_short"] = " ".join(what)[:300]

        if (what or why or how) and not summaries.get("summary_long"):
            summaries["summary_long"] = " ".join(what + why + how)[:1200]

        # Capability tags: from key_deps and/or tags if present
        cap_tags = _as_list(ai_json.get("capability_tags")) or _as_list(
            ai_json.get("key_deps")
        )
        if cap_tags:
            existing_ct = summaries.get("capability_tags") or []
            if not isinstance(existing_ct, list):
                existing_ct = [str(existing_ct)]
            summaries["capability_tags"] = sorted(
                {str(x) for x in (existing_ct + cap_tags)}
            )

        # --- Contracts: public_api + risks ---
        ai_public = _as_list(ai_json.get("public_api"))
        existing_public = contracts.get("public_api") or []
        if not isinstance(existing_public, list):
            existing_public = [str(existing_public)]
        if ai_public:
            contracts["public_api"] = sorted(
                {str(x) for x in (existing_public + ai_public)}
            )

        risks = _as_list(ai_json.get("risks"))
        if risks:
            existing_risks = contracts.get("risks_gotchas") or []
            if not isinstance(existing_risks, list):
                existing_risks = [str(existing_risks)]
            contracts["risks_gotchas"] = existing_risks + risks

        # --- Key dependencies (top-level convenience) ---
        key_deps = _as_list(ai_json.get("key_deps"))
        if key_deps:
            existing_kd = meta.get("key_deps") or []
            if not isinstance(existing_kd, list):
                existing_kd = [str(existing_kd)]
            meta["key_deps"] = sorted({str(x) for x in (existing_kd + key_deps)})

        # --- Assumptions / invariants ---
        assumptions = _as_list(ai_json.get("assumptions_invariants"))
        if assumptions:
            existing_ai = contracts.get("assumptions_invariants") or []
            if not isinstance(existing_ai, list):
                existing_ai = [str(existing_ai)]
            contracts["assumptions_invariants"] = existing_ai + assumptions

        # --- I/O contracts (optional passthrough) ---
        io_contracts = ai_json.get("io_contracts")
        if isinstance(io_contracts, dict):
            # Store raw, but keep it small and stringified.
            contracts["io_contracts"] = {
                k: _as_list(v)
                for k, v in io_contracts.items()
                if k in {"inputs", "outputs", "errors", "side_effects"}
            }

        # --- Metrics: test notes ---
        tests = _as_list(ai_json.get("tests"))
        if tests:
            existing_notes = metrics.get("test_notes") or []
            if not isinstance(existing_notes, list):
                existing_notes = [str(existing_notes)]
            metrics["test_notes"] = existing_notes + tests

        meta["summaries"] = summaries
        meta["contracts"] = contracts
        meta["metrics"] = metrics

    # ---------- Card index IO ----------
    def load_card_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load .aidev/cards/index.json, normalizing summary fields and staleness flags.

        If index.json is missing or corrupt, attempt to rebuild from existing
        per-file *.card.json files to preserve AI summaries and avoid unnecessary
        LLM refresh work.
        """
        data: Dict[str, Dict[str, Any]] = {}

        # 1) Try the index file first
        try:
            if self._cards_index_path.exists():
                raw = json.loads(self._cards_index_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(raw, dict):
                    data = raw
                else:
                    data = {}
        except Exception:
            data = {}

        # 2) If empty, rebuild from per-file cards
        if not data:
            data = self._rebuild_index_from_cards()

        # 3) Normalize entries
        try:
            if isinstance(data, dict):
                for rel, meta in list(data.items()):
                    if not isinstance(meta, dict):
                        data[rel] = {}
                        continue

                    s = self._ensure_summary_obj(meta)

                    meta.setdefault("file_sha", "")
                    meta.setdefault("sha256", "")
                    meta.setdefault("kind", "")
                    meta.setdefault("size", 0)
                    meta.setdefault("chunks", [])

                    meta.setdefault("imports", [])
                    meta.setdefault("exports", [])
                    meta.setdefault("symbols", [])
                    meta.setdefault("todos", [])
                    meta.setdefault("routes", [])
                    meta.setdefault("cli_args", [])
                    meta.setdefault("env_vars", [])
                    meta.setdefault("owners", [])
                    meta.setdefault("git_last_ts", None)
                    meta.setdefault("git_last_author", None)
                    meta.setdefault("embedding", None)
                    meta.setdefault("language", "other")
                    meta.setdefault("imports_resolved", [])

                    fs = str(meta.get("file_sha") or "")
                    ai_sha = s.get("ai_sha")
                    has_ai = bool(s.get("ai_text") or s.get("ai_json"))

                    st = meta.get("staleness")
                    if not isinstance(st, dict):
                        st = {}

                    changed = bool(
                        st.get("changed")
                        or meta.get("sha_changed")
                        or meta.get("changed")
                    )

                    needs = st.get("needs_ai_refresh")
                    if needs is None:
                        legacy_needs = meta.get("needs_ai_refresh")
                        if legacy_needs is not None:
                            needs = bool(legacy_needs)
                        else:
                            needs = (not has_ai) or (fs and ai_sha and fs != ai_sha)
                    else:
                        needs = bool(needs)

                    meta["staleness"] = {
                        "changed": bool(changed),
                        "needs_ai_refresh": bool(needs),
                    }

                return data
        except Exception:
            pass

        return {}

    def _rebuild_index_from_cards(self) -> Dict[str, Dict[str, Any]]:
        """
        Best-effort rebuild of the in-memory cards index from existing per-file
        *.card.json files under .aidev/cards/.

        This is used when index.json is missing/corrupt. We prefer preserving any
        existing AI summary fields and staleness so we don't trigger unnecessary
        LLM refresh work later.
        """
        out: Dict[str, Dict[str, Any]] = {}

        try:
            cards_dir = self._cards_dir
            if not cards_dir.exists():
                return {}

            for p in cards_dir.rglob("*.card.json"):
                try:
                    # Compute rel path from filename:
                    # .aidev/cards/<rel>.card.json -> <rel>
                    rel_part = p.relative_to(cards_dir).as_posix()
                    if not rel_part.endswith(".card.json"):
                        continue
                    rel = rel_part[: -len(".card.json")]

                    # Skip if file no longer exists
                    fs_path = self.root / rel
                    if not fs_path.exists():
                        continue

                    payload = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        continue

                    summary = payload.get("summary") or {}
                    graph = payload.get("graph") or {}
                    st = payload.get("staleness") or {}

                    meta: Dict[str, Any] = {
                        "sha256": payload.get("sha256") or "",
                        "kind": payload.get("kind") or "",
                        "size": int(payload.get("size") or 0),
                        "chunks": payload.get("chunks") or [],
                        "file_sha": payload.get("file_sha") or "",
                        "language": payload.get("language") or _language_for_path(rel),

                        # enriched / graph-ish
                        "imports": graph.get("imports_raw") or [],
                        "imports_resolved": graph.get("imports_resolved") or [],
                        "exports": graph.get("exports") or [],
                        "symbols": graph.get("symbols") or [],
                        "routes": graph.get("routes") or [],
                        "cli_args": graph.get("cli_args") or [],
                        "env_vars": graph.get("env_vars") or [],

                        "owners": payload.get("owners") or [],
                        "git_last_ts": payload.get("git_last_ts"),
                        "git_last_author": payload.get("git_last_author"),
                        "contracts": payload.get("contracts") or {},
                        "embedding": payload.get("embedding"),

                        # summary (structured)
                        "summary": {
                            "heuristic": (summary.get("heuristic") or ""),
                            "ai_text": summary.get("ai_text"),
                            "ai_json": summary.get("ai_json"),
                            "ai_model": summary.get("ai_model"),
                            "ai_ts": summary.get("ai_ts"),
                            "ai_sha": summary.get("ai_sha"),
                        },

                        # optional richness
                        "summaries": payload.get("summaries"),
                        "metrics": payload.get("metrics"),
                        "key_deps": payload.get("key_deps"),
                        "role": payload.get("role"),

                        # staleness (will be normalized later too)
                        "staleness": {
                            "changed": bool(st.get("changed", False)),
                            "needs_ai_refresh": bool(st.get("needs_ai_refresh", False)),
                        },
                    }

                    # Normalize summary object shape
                    self._ensure_summary_obj(meta)

                    out[rel] = meta

                except Exception:
                    continue

        except Exception:
            return {}

        return out

    # ------ Preserve/Carry-Forward helpers for AI summaries ------

    def _load_existing_card_meta(self, rel: str) -> Dict[str, Any]:
        try:
            p = self._card_json_path(rel)
            if p.exists():
                meta = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    return meta
        except Exception:
            pass
        return {}
    def _preserve_ai_summary_fields(
        self,
        rel: str,
        meta: Dict[str, Any],
        cached: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Preserve AI summary fields when we did NOT just re-summarize this file.

        Rules:
        - If the incoming meta already has an AI summary (ai_text/ai_json),
          we keep it as-is and DO NOT overwrite it.
        - Otherwise, we try to carry forward AI fields from:
            1) the provided cached meta (from index.json), or
            2) as a fallback, an existing on-disk card (via _load_existing_card_meta),
          but only if ai_sha matches the current file_sha.
        """
        new_s = self._ensure_summary_obj(meta)
        file_sha = meta.get("file_sha")

        has_new_ai = bool(new_s.get("ai_text") or new_s.get("ai_json"))

        # Case 1: meta already has AI text/json -> leave it alone
        if has_new_ai:
            meta["summary"] = new_s
            return meta
        # --- Decide where to pull "old" AI from ---
        old_meta: Dict[str, Any] = {}

        # Preferred: cached meta from index.json (passed by update_cards)
        if isinstance(cached, dict) and cached:
            old_meta = cached
        else:
            # Fallback: load existing card JSON from disk (legacy behavior)
            try:
                existing = self._load_existing_card_meta(rel)
                if isinstance(existing, dict):
                    old_meta = existing
            except Exception:
                old_meta = {}

        old_s = self._ensure_summary_obj(old_meta) if old_meta else {}

        existing_ai_sha = old_s.get("ai_sha")
        existing_has_ai = bool(old_s.get("ai_text") or old_s.get("ai_json"))

        # Case 2: carry forward AI fields from previous meta/card IF SHA matches
        if (
            file_sha
            and existing_has_ai
            and existing_ai_sha
            and existing_ai_sha == file_sha
        ):
            for key in ("ai_text", "ai_json", "ai_model", "ai_ts", "ai_sha"):
                if key in old_s:
                    new_s[key] = old_s[key]

        meta["summary"] = new_s
        return meta

    def _make_contracts(
        self, rel: str, *, exports: List[str], symbols: List[str], env_vars: List[str]
    ) -> Dict[str, Any]:
        """
        Lightweight, heuristic contracts. Safe defaults; can be enriched later.
        """
        base = Path(rel).stem
        tests: List[str] = []

        # simple neighbor guesses
        for patt in (
            f"test_{base}.py",
            f"{base}_test.py",
            f"{base}.test.py",
            f"{base}.spec.ts",
            f"{base}.test.ts",
            f"test_{base}.ts",
            f"{base}.spec.js",
            f"{base}.test.js",
        ):
            for cand in self.structure.keys():
                if cand.endswith(patt):
                    tests.append(cand)
        tests = sorted(list(dict.fromkeys(tests)))[:12]

        return {
            "public_api": sorted(list(dict.fromkeys(exports or symbols))[:64]),
            "data_models": [],
            "io_contracts": [],
            "config_contracts": {"env_required": sorted(env_vars)[:64]},
            "compat_surface": sorted(list(dict.fromkeys(exports))[:64]),
            "assumptions_invariants": [],
            "test_neighbors": tests,
        }

    def _write_card_file(self, rel: str, meta: Dict[str, Any]) -> None:
        """
        Write per-file .card.json conforming to the v2 schema (no legacy mirrors).
        """
        s = self._ensure_summary_obj(meta)
        st = meta.get("staleness") or {}
        changed = bool(st.get("changed"))
        needs_ai_refresh = bool(st.get("needs_ai_refresh"))

        card_payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "path": rel,
            "kind": meta.get("kind") or "",
            "language": meta.get("language") or _language_for_path(rel),
            "size": int(meta.get("size") or 0),
            "sha256": meta.get("sha256") or "",
            "file_sha": meta.get("file_sha") or "",
            "chunks": meta.get("chunks") or [],
            "summary": {
                "heuristic": s.get("heuristic") or "",
                "ai_text": s.get("ai_text"),
                "ai_json": s.get("ai_json"),
                "ai_model": s.get("ai_model"),
                "ai_ts": s.get("ai_ts"),
                "ai_sha": s.get("ai_sha"),
            },
            "graph": {
                "imports_raw": meta.get("imports") or [],
                "imports_resolved": meta.get("imports_resolved") or [],
                "exports": meta.get("exports") or [],
                "symbols": meta.get("symbols") or [],
                "routes": meta.get("routes") or [],
                "cli_args": meta.get("cli_args") or [],
                "env_vars": meta.get("env_vars") or [],
            },
            "owners": meta.get("owners") or [],
            "git_last_ts": meta.get("git_last_ts"),
            "git_last_author": meta.get("git_last_author"),
            "contracts": meta.get("contracts")
            or self._make_contracts(
                rel,
                exports=meta.get("exports") or [],
                symbols=meta.get("symbols") or [],
                env_vars=meta.get("env_vars") or [],
            ),
            "embedding": meta.get("embedding"),
            "staleness": {
                "changed": changed,
                "needs_ai_refresh": needs_ai_refresh,
            },
        }

        # --- v2+ optional richness: summaries/metrics/key_deps (and friends) ---
        summaries = meta.get("summaries")
        if isinstance(summaries, dict) and summaries:
            card_payload["summaries"] = summaries

        metrics = meta.get("metrics")
        if isinstance(metrics, dict) and metrics:
            card_payload["metrics"] = metrics

        key_deps = meta.get("key_deps")
        if isinstance(key_deps, list) and key_deps:
            card_payload["key_deps"] = key_deps

        role = meta.get("role")
        if isinstance(role, dict) and role:
            # Only write if your schema includes 'role'; harmless otherwise but
            # may trigger a validation warning if omitted from cards.schema.json.
            card_payload["role"] = role

        try:
            v = self._get_card_validator()
            if v is not None:
                for err in v.iter_errors(card_payload):
                    logger.warning(
                        "Card JSON schema violation",
                        ctx={"path": rel, "error": err.message},
                    )
        except Exception as e:
            logger.warning(
                "Card validation failed",
                ctx={"path": rel, "err": str(e)[:200]},
            )

        try:
            p = self._card_json_path(rel)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(card_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(p)

        except Exception as e:
            logger.warning(
                "Card write failed",
                ctx={"path": rel, "err": str(e)[:200]},
            )

    def _prune_deleted_files_from_index(
        self, idx: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Remove card/index entries for files that no longer exist in the project
        (or are no longer present in the current structure), and delete their
        per-file .card.json files.

        This keeps .aidev/cards/* and cards/index.json in sync with the actual
        project tree and prevents deleted files from leaking into project_map.
        """
        try:
            existing: Optional[set[str]] = None
            try:
                if isinstance(self.structure, dict):
                    existing = set(self.structure.keys())
            except Exception:
                existing = None

            for rel in list(idx.keys()):
                # If we have a structure map, treat anything not in it as removed.
                missing_in_structure = existing is not None and rel not in existing

                # Also check the filesystem as a safety net.
                p = self.root / rel
                missing_on_disk = not p.exists()

                if missing_in_structure or missing_on_disk:
                    idx.pop(rel, None)
                    try:
                        # Best-effort delete of per-file card JSON.
                        safe_rel = Path(rel).as_posix()
                        card_path = self._cards_dir / f"{safe_rel}.card.json"
                        if card_path.exists():
                            card_path.unlink()
                    except Exception:
                        # Ignore removal errors; we just don't want stale cards.
                        pass
        except Exception:
            # Defensive: never break card saving on prune problems.
            pass

    def save_card_index(self, idx: Dict[str, Dict[str, Any]]) -> None:
        """
        Persist the cards index AND per-file .card.json files.
        """
        # First, prune any denied entries (and delete their per-file card JSONs)
        try:
            denied = [
                rel
                for rel in list(idx.keys())
                if _matches_any(_to_posix(rel), _INTERNAL_DENY_GLOBS)
            ]
            for rel in denied:
                idx.pop(rel, None)
                try:
                    p = self._card_json_path(rel)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        # Prune entries for deleted / no-longer-structured files
        self._prune_deleted_files_from_index(idx)

        # Optional: sample a few entries before we start mutating them
        for rel, meta in list(idx.items())[:5]:
            pass

        # Carry forward AI fields (already handled pre-save) and compute staleness.
        for rel, meta in idx.items():
            if not isinstance(meta, dict):
                continue

            s = self._ensure_summary_obj(meta)

            fs = str(meta.get("file_sha") or "")
            ai_sha = s.get("ai_sha")
            has_ai = bool(s.get("ai_text") or s.get("ai_json"))

            st = meta.get("staleness")
            if not isinstance(st, dict):
                st = {}

            # changed: prefer existing flag, fall back to legacy
            if "changed" in st:
                changed = bool(st.get("changed"))
            else:
                changed = bool(meta.get("sha_changed") or meta.get("changed"))

            # needs_ai_refresh: recompute from file_sha vs ai_sha + has_ai
            if fs and ai_sha:
                needs_ai_refresh = (fs != ai_sha)
            else:
                needs_ai_refresh = not has_ai

            meta["staleness"] = {
                "changed": changed,
                "needs_ai_refresh": needs_ai_refresh,
            }

            # Drop legacy flags on write; staleness is canonical in v2
            meta.pop("sha_changed", None)
            meta.pop("changed", None)
            meta.pop("needs_ai_refresh", None)

            # Mirror contracts.public_api to top-level convenience key for downstream tools
            try:
                if not meta.get("public_api"):
                    pa = (meta.get("contracts") or {}).get("public_api") or []
                    if isinstance(pa, list):
                        meta["public_api"] = pa[:64]
            except Exception:
                pass

            # Per-file card
            self._write_card_file(rel, meta)

        # Write index.json (store full meta; readers can pick what they want)
        try:
            tmp = self._cards_index_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(idx, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._cards_index_path)
        except Exception as e:
            logger.warning("Index write failed", ctx={"err": str(e)[:200]})

    def _load_structure_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the structure cache and normalize it into the
        {rel_path: info_dict} shape expected by update_cards().

        Supported legacy shapes:
        - {"files": {<rel>: {...}}}
        - {<rel>: {...}}
        - [{"path": "...", ...}, ...] / [{"rel": "...", ...}, ...]
        Anything else falls back to {}.
        """
        p = getattr(self, "_structure_cache_path", None)
        if not p:
            return {}

        try:
            if not p.exists():
                return {}
            raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}

        try:
            # Newer/alternate shape: {"files": {rel: info}}
            if isinstance(raw, dict):
                files_obj = raw.get("files")

                if isinstance(files_obj, dict):
                    return {
                        str(rel): (meta if isinstance(meta, dict) else {})
                        for rel, meta in files_obj.items()
                        if isinstance(rel, str) and rel.strip()
                    }

                # If "files" exists but is not a dict, treat as invalid shape.
                if "files" in raw and not isinstance(files_obj, dict):
                    return {}

                # Dict-of-dicts shape: {rel: info}
                # Filter out non-dict values entirely to avoid polluting with metadata keys.
                out: Dict[str, Dict[str, Any]] = {}
                for rel, meta in raw.items():
                    if not isinstance(rel, str) or not rel.strip():
                        continue
                    if not isinstance(meta, dict):
                        continue
                    out[str(rel)] = meta
                return out

            # Legacy list-of-objects shape
            if isinstance(raw, list):
                out: Dict[str, Dict[str, Any]] = {}
                for item in raw:
                    if not isinstance(item, dict):
                        continue

                    rel = (
                        item.get("rel")
                        or item.get("path")
                        or item.get("file")
                        or item.get("name")
                    )
                    if not isinstance(rel, str) or not rel.strip():
                        continue

                    out[rel.strip()] = dict(item)
                return out

        except Exception:
            return {}

        return {}

    # ---------- NEW: helpers for "changed" and graph importance ----------
    def _ensure_fresh_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Run a cheap structural pass to ensure the card index is in sync with current files,
        without re-running AI summaries.
        """
        # If index is missing/corrupt, load_card_index will now rebuild from cards.
        _ = self.load_card_index()
        idx = self.update_cards(changed_only=True, force=False)
        return idx

    def _list_needs_ai_refresh(
        self, idx: Dict[str, Dict[str, Any]], paths: Optional[List[str]] = None
    ) -> List[str]:
        if paths:
            return sorted(paths)

        targets: List[str] = []
        for rel, meta in idx.items():
            if not isinstance(meta, dict):
                continue
            st = meta.get("staleness") or {}
            s = self._ensure_summary_obj(meta)
            has_ai = bool(s.get("ai_text") or s.get("ai_json"))
            needs_flag = bool(st.get("needs_ai_refresh"))
            if needs_flag or not has_ai:
                targets.append(rel)

        return sorted(targets)

    def _load_graph(self) -> Dict[str, Any]:
        """
        Synthesize graph from the cards index; fall back to legacy file if present.
        """
        # Preferred: build from cards (authoritative source)
        try:
            idx = self.load_card_index()
            if isinstance(idx, dict) and idx:
                nodes = sorted(idx.keys())
                edges = {
                    rel: list((meta or {}).get("imports") or [])
                    for rel, meta in idx.items()
                }
                edges_resolved = {
                    rel: list((meta or {}).get("imports_resolved") or [])
                    for rel, meta in idx.items()
                }
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "edges_resolved": edges_resolved,
                }
        except Exception:
            pass

        # Legacy fallback: use the old file if it exists
        try:
            if self._graph_index_path.exists():
                return json.loads(
                    self._graph_index_path.read_text(encoding="utf-8")
                )
        except Exception:
            pass
        return {"nodes": [], "edges": {}, "edges_resolved": {}}

    # ---------- Card graph access / related-file helpers ----------
    def get_card(self, rel: str) -> Dict[str, Any]:
        """
        Convenience: return the card meta for a given relative path, or {}.

        Uses the cards index as the single source of truth.
        """
        idx = self.load_card_index()
        meta = idx.get(rel) or {}
        if isinstance(meta, dict):
            # normalize summary structure for callers
            self._ensure_summary_obj(meta)
            return meta
        return {}

    def _compute_reverse_imports(
        self, idx: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, List[str]]:
        """
        Build a mapping of { target_rel -> [importer_rel, ...] } from imports_resolved.

        This is intentionally cheap and run on-demand; if you want to cache it
        across calls, do that in a higher-level helper/module.
        """
        if idx is None:
            idx = self.load_card_index()

        rev: Dict[str, List[str]] = {}
        for src, meta in (idx or {}).items():
            if not isinstance(meta, dict):
                continue
            for tgt in meta.get("imports_resolved") or []:
                if not isinstance(tgt, str):
                    continue
                rev.setdefault(tgt, []).append(src)
        # de-dup & sort for stability
        for tgt, srcs in rev.items():
            seen: set[str] = set()
            uniq: List[str] = []
            for s in srcs:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            uniq.sort()
            rev[tgt] = uniq
        return rev

    def get_related_files(
        self,
        rel: str,
        *,
        max_same_dir: int = 8,
        max_dependencies: int = 16,
        max_dependents: int = 16,
        max_tests: int = 8,
    ) -> Dict[str, List[str]]:
        """
        Return a structured view of "nearby" files for a target path:

            {
                "same_dir": [...],
                "dependencies": [...],   # what this file imports
                "dependents": [...],     # who imports this file
                "tests": [...],          # associated tests from contracts + heuristics
            }

        This is intentionally lightweight and side-effect free; it uses the
        existing cards index + structure map and never mutates disk.
        """
        idx = self.load_card_index()
        meta = idx.get(rel) or {}
        self._ensure_summary_obj(meta)

        # 1) Same-directory siblings (based on the current structure)
        same_dir: List[str] = []
        try:
            rel_path = Path(rel)
            parent = rel_path.parent
            for other in self.structure.keys():
                if other == rel:
                    continue
                if Path(other).parent == parent:
                    same_dir.append(other)
        except Exception:
            same_dir = []

        same_dir = sorted(dict.fromkeys(same_dir))[: max_same_dir]

        # 2) Dependencies (resolved imports from this file)
        deps = list(meta.get("imports_resolved") or [])
        deps = sorted(dict.fromkeys([d for d in deps if isinstance(d, str)]))[
            : max_dependencies
        ]

        # 3) Dependents (reverse edges: who imports this file)
        rev = self._compute_reverse_imports(idx)
        dependents = sorted(dict.fromkeys(rev.get(rel, [])))[: max_dependents]

        # 4) Tests: prefer contracts.test_neighbors, then fallback heuristics.
        tests: List[str] = []
        contracts = meta.get("contracts") or {}
        if isinstance(contracts, dict):
            tn = contracts.get("test_neighbors") or []
            if isinstance(tn, list):
                tests.extend([t for t in tn if isinstance(t, str)])

        # simple heuristic: any sibling file whose name looks like a test
        if len(tests) < max_tests:
            base = Path(rel).stem
            test_suffixes = (
                f"test_{base}.py",
                f"{base}_test.py",
                f"{base}.test.py",
                f"{base}.spec.ts",
                f"{base}.test.ts",
                f"test_{base}.ts",
                f"{base}.spec.js",
                f"{base}.test.js",
            )
            for other in same_dir:
                name = os.path.basename(other)
                if name in test_suffixes:
                    tests.append(other)

        tests = sorted(dict.fromkeys(tests))[: max_tests]

        return {
            "same_dir": same_dir,
            "dependencies": deps,
            "dependents": dependents,
            "tests": tests,
        }

    def get_top_cards_for_analyze(
        self,
        top_k: int = 20,
        query: Optional[str] = None,
        *,
        filter_includes: Optional[List[str]] = None,
        filter_excludes: Optional[List[str]] = None,
        exclude_generated: bool = True,
        max_summary_len: int = 800,
    ) -> List[Dict[str, Any]]:
        """Return deterministic, deduped top-N cards for Analyze payload construction.

        Determinism guarantees (for the same repo state and same inputs):
        - Candidate pool is selected via KnowledgeBase.select(...) with stable sorting.
        - Scoring uses existing select_cards() logic (unchanged).
        - Results are de-duplicated by path/id and then sorted by (-score, path).

        Size guarantees:
        - Returned dicts are minimal and never include embeddings or large blobs.
        - Each summary is capped to max_summary_len characters.

        Note: select_cards() only applies the internal denylist when filters are
        provided; Analyze must always exclude generated/internal artifacts, so
        this helper enforces _INTERNAL_DENY_GLOBS explicitly when requested.
        """

        # Defensive normalization
        try:
            k = int(top_k)
        except Exception:
            k = 20
        if k <= 0:
            return []

        q = (query or "").strip() or "project"

        inc = list(filter_includes or [])
        exc = list(filter_excludes or [])
        if exclude_generated:
            # Ensure internal denylist is always honored for Analyze.
            exc = list(dict.fromkeys(exc + _INTERNAL_DENY_GLOBS))

        # Use select() to build a stable candidate pool with denylist applied.
        candidates = self.select(inc, exc)
        if not candidates:
            return []

        # Ask for more than top_k to allow deterministic dedupe/trimming.
        max_k_candidates = min(len(candidates), max(k * 3, k))

        # Score using existing logic, but constrain to our candidate pool.
        scored = self.select_cards(
            q,
            top_k=max_k_candidates,
            filter_includes=inc,
            filter_excludes=exc,
        )

        # Deterministic de-dupe by path/id preserving first seen.
        by_path: Dict[str, Dict[str, Any]] = {}
        for item in scored:
            if not isinstance(item, dict):
                continue
            rel = item.get("path") or item.get("id")
            if not isinstance(rel, str) or not rel:
                continue
            if rel in by_path:
                continue
            by_path[rel] = item

        # Normalize, trim, and keep minimal fields.
        out: List[Dict[str, Any]] = []
        for rel, item in by_path.items():
            try:
                score = float(item.get("score") or 0.0)
            except Exception:
                score = 0.0

            summary = item.get("summary")
            if not isinstance(summary, str):
                summary = ""
            summary = " ".join(summary.split())
            if max_summary_len and len(summary) > max_summary_len:
                summary = summary[: max(0, max_summary_len - 1)].rstrip() + "…"

            # Pull small metadata from the index (avoid large fields).
            size_val: Optional[int] = None
            lang_val: str = "other"
            try:
                meta = self.get_card(rel)
                if isinstance(meta, dict):
                    try:
                        size_val = int(meta.get("size") or 0)
                    except Exception:
                        size_val = None
                    lang_val = str(meta.get("language") or "other")
            except Exception:
                size_val = None
                lang_val = "other"

            out.append(
                {
                    "id": rel,
                    "path": rel,
                    "summary": summary,
                    "score": float(score),
                    "size": size_val,
                    "language": lang_val or "other",
                }
            )

        # Stable ordering and final truncation.
        out.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("path") or "")))
        return out[:k]

    def _importance_scores(self, idx: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        graph = self._load_graph()
        edges_resolved: Dict[str, List[str]] = graph.get("edges_resolved") or {}
        indeg: Dict[str, int] = {}
        outdeg: Dict[str, int] = {}

        for a, outs in (edges_resolved or {}).items():
            outdeg[a] = len(outs or [])
            for b in outs or []:
                indeg[b] = indeg.get(b, 0) + 1

        scores: List[Tuple[str, float]] = []
        for rel, meta in idx.items():
            base = 0.0
            name = rel.lower()
            routes = meta.get("routes") or []
            cli = meta.get("cli_args") or []
            exports = meta.get("exports") or []
            symbols = meta.get("symbols") or []
            kind = (meta.get("kind") or "").lower()
            sz = int(meta.get("size") or 0)

            if any(
                x in name
                for x in ("/main.", "/app.", "/server.", "/index.", "__init__.py")
            ):
                base += 3.0
            if routes:
                base += min(5.0, 1.0 + 0.5 * len(routes))
            if cli:
                base += min(3.0, 0.5 * len(cli) + 1.0)
            base += min(3.0, 0.05 * (len(exports) + len(symbols)))
            base += 0.6 * float(indeg.get(rel, 0)) + 0.25 * float(outdeg.get(rel, 0))
            if sz > 80_000:
                base -= 1.0
            if "config" in kind or name.endswith((".json", ".yaml", ".yml")):
                base += 0.5
            scores.append((rel, base))

        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores

    # ---------- Card generation / update ----------
    def update_cards(
        self,
        files: Optional[List[str]] = None,
        *,
        force: bool = False,
        changed_only: bool = True,
        max_bytes_per_file: int = 100_000,
        write_graph_index: Optional[bool] = None,
        compute_embeddings: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate/update .aidev/cards/index.json and per-file .card.json with summaries and rich metadata.
        """
        idx = self.load_card_index()
        struct_idx = self._load_structure_cache()

        # Embeddings OFF by default for structural pass
        if compute_embeddings is None:
            compute_embeddings = False

        # By default, do NOT write the legacy graph file. Can be re-enabled via env.
        if write_graph_index is None:
            write_graph_index = _env_bool("AIDEV_WRITE_GRAPH_INDEX", False)

        if not files:
            files = sorted(self.structure.keys())

        # Always skip internal/generated content defensively
        files = [
            rel
            for rel in files
            if not _matches_any(_to_posix(rel), _INTERNAL_DENY_GLOBS)
        ]

        tsconfig = _load_tsconfig(self.root)
        owners_rules = _parse_codeowners(self.root)

        nodes: List[str] = []
        edges: Dict[str, List[str]] = {}
        edges_resolved: Dict[str, List[str]] = {}

        for rel in files:
            info = struct_idx.get(rel) or {}
            struct_sha = str(info.get("sha256") or "")
            kind = str(info.get("kind") or self.structure.get(rel) or "other")
            size = int(info.get("size") or 0)
            chunks = info.get("chunks") or []

            cached = idx.get(rel) or {}
            # Make sure cached has a normalized summary object we can merge from

            can_skip_read = (
                (not force)
                and cached
                and cached.get("sha256") == struct_sha
                and cached.get("kind") == kind
                and int(cached.get("size") or 0) == size
                and str(cached.get("file_sha") or "") != ""
            )

            new_text: Optional[str]
            if can_skip_read:
                new_text = None
                new_file_sha = str(cached.get("file_sha"))
            else:
                try:
                    with open(self.root / rel, "rb") as f:
                        raw = f.read(max_bytes_per_file)
                        _ = f.read(1)
                    new_text = raw.decode("utf-8", errors="replace")
                except Exception:
                    new_text = ""
                normalized = _normalize_for_hash(new_text or "")
                new_file_sha = _sha256_text(normalized)

            prev_file_sha = str(cached.get("file_sha") or "")
            file_changed = new_file_sha != prev_file_sha
            must_recalc_summary = (
                force
                or (not changed_only)
                or file_changed
                or (not cached)
                or (not cached.get("summary"))
            )

            # Enriched metadata (from cache if we skipped reading)
            imports = list(cached.get("imports") or [])
            exports = list(cached.get("exports") or [])
            symbols = list(cached.get("symbols") or [])
            todos = list(cached.get("todos") or [])
            routes = list(cached.get("routes") or [])
            cli_args = list(cached.get("cli_args") or [])
            env_vars = list(cached.get("env_vars") or [])
            owners = list(cached.get("owners") or [])
            git_last_ts = cached.get("git_last_ts")
            git_last_author = cached.get("git_last_author")
            embedding = cached.get("embedding")
            language = cached.get("language") or _language_for_path(rel)

            imports_resolved: List[str] = list(cached.get("imports_resolved") or [])

            if new_text is not None:
                text_for_extract = new_text
                imports = _extract_imports(text_for_extract)
                exports, symbols = _extract_exports_and_symbols(text_for_extract, rel)
                todos = _extract_todos(text_for_extract)
                routes = _extract_routes(text_for_extract)
                cli_args = _extract_cli_args(text_for_extract)
                env_vars = _extract_env_vars(text_for_extract)
                owners = _owners_for_path(rel, owners_rules) if owners_rules else []
                git_last_ts, git_last_author = _git_last_change(self.root, rel)
                language = _language_for_path(rel)

            if must_recalc_summary:
                if new_text is None:
                    try:
                        with open(self.root / rel, "rb") as f:
                            raw = f.read(max_bytes_per_file)
                            _ = f.read(1)
                        new_text = raw.decode("utf-8", errors="replace")
                    except Exception:
                        new_text = ""
                new_heuristic = _heuristic_summary(
                    new_text or "", rel, imports, exports, symbols
                )
            else:
                s_cached = self._ensure_summary_obj(cached)
                new_heuristic = (
                    s_cached.get("heuristic")
                    or (
                        cached.get("summary")
                        if isinstance(cached.get("summary"), str)
                        else ""
                    )
                    or ""
                )

            # Resolve imports to project files
            if new_text is not None:
                resolved: List[str] = []
                from_path = self.root / rel
                for spec in imports:
                    rp = _resolve_import_specifier(spec, from_path, self.root, tsconfig)
                    if rp:
                        resolved.append(rp)
                imports_resolved = resolved

            # Contracts (cheap/heuristic)
            contracts = self._make_contracts(
                rel, exports=exports, symbols=symbols, env_vars=env_vars
            )

            # Start from cached to preserve AI + embedding etc.,
            # then overwrite structural/enriched fields.
            meta = dict(cached)
            meta.update(
                {
                    "sha256": struct_sha,
                    "kind": kind,
                    "size": size,
                    "chunks": chunks,
                    "file_sha": new_file_sha,
                    "language": language,
                    # enriched metadata
                    "imports": imports,
                    "exports": exports,
                    "symbols": symbols,
                    "todos": todos,
                    "routes": routes,
                    "cli_args": cli_args,
                    "env_vars": env_vars,
                    "owners": owners,
                    "git_last_ts": git_last_ts,
                    "git_last_author": git_last_author,
                    "embedding": embedding,
                    "imports_resolved": imports_resolved,
                    "contracts": contracts,
                    "staleness": {
                        "changed": bool(file_changed),
                        # needs_ai_refresh will be recomputed in save_card_index
                    },
                }
            )

            # Ensure we have a summary object and set the heuristic part
            s_new = self._ensure_summary_obj(meta)
            s_new["heuristic"] = new_heuristic or ""
            meta["summary"] = s_new

            # Preserve AI summary fields (if any) from cached meta
            meta = self._preserve_ai_summary_fields(rel, meta, cached)

            idx[rel] = meta
            # Graph aggregation
            nodes.append(rel)
            edges[rel] = list(imports)
            edges_resolved[rel] = list(imports_resolved)

        # Persist cards (and per-file card JSONs)
        self.save_card_index(idx)

        return idx

    def _summarize_one_file(
        self,
        rel: str,
        meta: Dict[str, Any],
        *,
        model: Optional[str],
        max_bytes_per_file: int,
        compute_embeddings: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        result: Dict[str, Any] = {
            "path": rel,
            "ok": False,
            "summary_len": 0,
            "error": None,
        }

        file_sha = str(meta.get("file_sha") or "")
        s = self._ensure_summary_obj(meta)
        ai_sha = s.get("ai_sha")
        has_ai = bool(s.get("ai_text") or s.get("ai_json"))

        # If we already have an AI summary for this exact file SHA, skip.
        if has_ai and ai_sha and file_sha and (file_sha == ai_sha):
            result["ok"] = True
            result["summary_len"] = len(s.get("ai_text") or "")
            return meta, result

        # Read (truncated) file text
        try:
            with open(self.root / rel, "rb") as f:
                raw_bytes = f.read(max_bytes_per_file)
                _ = f.read(1)  # nudge truncated detection / future use
            text = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        normalized = _normalize_for_hash(text or "")

        # Lightweight structural context
        context = {
            "rel_path": rel,
            "kind": meta.get("kind") or self.structure.get(rel) or "source",
            "size": int(meta.get("size") or 0),
            "imports": meta.get("imports") or [],
            "exports": meta.get("exports") or [],
            "symbols": meta.get("symbols") or [],
            "routes": meta.get("routes") or [],
            "cli_args": meta.get("cli_args") or [],
            "env_vars": meta.get("env_vars") or [],
        }

        ai_text: Optional[str] = None
        ai_json: Optional[Dict[str, Any]] = None
        err_msg: Optional[str] = None

        try:
            # IMPORTANT: call the LLMClient instance method summarize_file_card(...)
            # We pass a stage='card_summarize' so the llm_client can resolve a per-stage model.
            # We fall back to calling without a stage kwarg if the client doesn't accept it,
            # preserving backward compatibility.
            from .llm_client import LLMClient  # type: ignore

            client = LLMClient(model=model)

            raw = None
            try:
                raw = client.summarize_file_card(
                    rel_path=rel,
                    text=normalized,
                    max_tokens=4096,
                    schema=None,
                    stage="card_summarize",
                )
            except TypeError:
                # Older LLMClient may not accept 'stage' yet; call without it.
                raw = client.summarize_file_card(
                    rel_path=rel,
                    text=normalized,
                    max_tokens=4096,
                    schema=None,
                )

            # Defensive extraction of model metadata if the client returned it.
            model_from_response: Optional[str] = None
            try:
                if isinstance(raw, tuple) and len(raw) > 1 and isinstance(raw[1], dict):
                    model_from_response = raw[1].get("model")
                elif isinstance(raw, dict):
                    model_from_response = raw.get("model")
            except Exception:
                model_from_response = None

            # ---- Unwrap (data, res) tuple if needed (kept for back-compat) ----
            primary = raw
            if isinstance(primary, tuple) and primary:
                primary = primary[0]

            parsed_json: Optional[Dict[str, Any]] = None
            card_wrapper: Optional[Dict[str, Any]] = None

            # primary can be:
            # - { "what": ..., "why": ..., ... }
            # - { "path": ..., "ai_summary": { ... }, "title": ..., "lines": [...] }
            # - already-string JSON (less likely, but handle it)
            if isinstance(primary, dict):
                if isinstance(primary.get("ai_summary"), dict):
                    card_wrapper = primary
                    parsed_json = primary["ai_summary"]
                else:
                    parsed_json = primary
            elif isinstance(primary, str):
                text_val = primary.strip()
                if text_val:
                    try:
                        candidate = json.loads(text_val)
                        if isinstance(candidate, dict):
                            parsed_json = candidate
                        else:
                            ai_text = text_val
                    except Exception:
                        ai_text = text_val

            ctx: Dict[str, Any] = {
                "path": rel,
                "primary_type": type(primary).__name__,
                "has_ai_json": parsed_json is not None,
                "has_card_wrapper": isinstance(card_wrapper, dict),
            }
            if isinstance(parsed_json, dict):
                ctx["ai_json_keys"] = list(parsed_json.keys())
            preview = ""
            if isinstance(ai_text, str):
                preview = ai_text[:200]
            if preview:
                ctx["ai_text_preview"] = preview

            # If we got structured JSON, turn it into ai_text + ai_json
            if parsed_json is not None:
                ai_json = parsed_json

                # Best-effort schema validation (optional)
                try:
                    v = self._get_ai_validator()
                    if v is not None:
                        for err in v.iter_errors(ai_json):
                            logger.warning(
                                "AI summary JSON schema violation",
                                ctx={"path": rel, "error": err.message},
                            )
                except Exception as ve:
                    logger.warning(
                        "AI summary JSON validation failed",
                        ctx={"path": rel, "err": str(ve)[:200]},
                    )

                def _take(val: Any, n: int = 6) -> List[str]:
                    out: List[str] = []
                    if isinstance(val, str):
                        s2 = val.strip()
                        if s2:
                            out.append(s2)
                    elif isinstance(val, list):
                        for item in val[:n]:
                            if isinstance(item, str):
                                s3 = item.strip()
                                if s3:
                                    out.append(s3)
                            else:
                                s4 = str(item).strip()
                                if s4:
                                    out.append(s4)
                    return out

                parts: List[str] = []
                parts += _take(ai_json.get("what"), 1)
                parts += _take(ai_json.get("why"), 1)
                parts += _take(ai_json.get("how"), 1)
                parts += _take(ai_json.get("public_api"), 6)
                parts += _take(ai_json.get("key_deps"), 6)
                parts += _take(ai_json.get("risks"), 6)
                parts += _take(ai_json.get("assumptions_invariants"), 4)
                parts += _take(ai_json.get("tests"), 6)
                parts += _take(ai_json.get("notes"), 1)

                # If everything above was empty but we had a wrapper, fall back
                if not parts and isinstance(card_wrapper, dict):
                    lines = card_wrapper.get("lines")
                    if isinstance(lines, list):
                        parts = [str(x).strip() for x in lines if str(x).strip()]
                    if not parts and isinstance(card_wrapper.get("title"), str):
                        t = card_wrapper["title"].strip()
                        if t:
                            parts = [t]

                if parts:
                    ai_text = "; ".join(parts).strip()[:1200] or ai_text
            else:
                ai_text = ai_text or None

        except Exception as e:
            err_msg = str(e)[:400]
            logger.warning(
                "AI summarize failed",
                ctx={"path": rel, "err": err_msg},
            )

        # Choose the model name we’ll record in the card
        # Prefer a model returned by the LLMClient response (model_from_response),
        # else prefer an explicit 'model' argument, then environment fallbacks.
        try:
            effective_model_from_response = locals().get("model_from_response")
        except Exception:
            effective_model_from_response = None

        model_name = (
            effective_model_from_response
            or model
            or os.getenv("OPENAI_MODEL")
        )

        # ---- Commit into meta if we got anything useful ----
        if ai_text or ai_json:
            try:
                if ai_json is not None:
                    try:
                        self._apply_structured_summary(meta, ai_json)
                    except Exception as e:
                        logger.warning(
                            "apply_structured_summary failed",
                            ctx={"path": rel, "err": str(e)[:200]},
                        )

                # IMPORTANT: use the returned meta so callers always see the updated dict
                meta = self._set_ai_summary(
                    meta,
                    ai_text=ai_text,
                    ai_json=ai_json,
                    model=model_name,
                    file_sha=file_sha,
                )

                # Ensure staleness reflects that this file is now fresh.
                st = meta.setdefault("staleness", {})
                st["needs_ai_refresh"] = False

                result["ok"] = True
                result["summary_len"] = len(ai_text or "")

                # Optional: embed summary text
                if compute_embeddings and _llm_embed_texts is not None:
                    try:
                        src = ai_text or self._effective_summary_text(meta)
                        if src:
                            vecs = _llm_embed_texts([src]) or []
                            if vecs:
                                meta["embedding"] = vecs[0]
                    except Exception:
                        # Embedding is a nice-to-have; don't fail whole summary
                        pass

            except Exception as e:
                err_msg = str(e)[:400]
                result["error"] = err_msg
        else:
            result["error"] = err_msg or "No summary produced."

        return meta, result

    # ---------- LLM AI summaries (changed-only by hashes) ----------
    def generate_ai_summaries(
        self,
        *,
        changed_only: bool = True,
        paths: Optional[List[str]] = None,
        model: Optional[str] = "gpt-5-mini",
        max_bytes_per_file: int = 120_000,
        compute_embeddings: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Back-compat wrapper around summarize_changed.

        Returns the updated index (as before), but internally uses the
        newer summarize_changed() logic.
        """
        self.summarize_changed(
            paths=paths if not changed_only else None,
            model=model,
            max_bytes_per_file=max_bytes_per_file,
            compute_embeddings=compute_embeddings,
        )
        # We ignore report here and return a fresh index snapshot like the old API.
        return self._ensure_fresh_index()

    # ---------- Structured "changed" flow with per-file results ----------
    def summarize_changed(
        self,
        *,
        paths: Optional[List[str]] = None,
        model: Optional[str] = "gpt-5-mini",
        max_bytes_per_file: int = 120_000,
        compute_embeddings: Optional[bool] = None,
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run AI summaries for changed files (or provided paths) and return a UI-friendly report.

        Returns a dict shaped like:

            {
              "ok": bool,
              "total": int,
              "updated": int,
              "failed": int,
              "skipped": int,
              "results": [
                { "path": str, "ok": bool, "summary_len": int, "error": str | None },
                ...
              ],
              "message": str,
              "error": str | None,
            }
        """
        # Concurrency cap for per-file LLM summaries.
        # Prefer config, then env, default 5; clamp to int >= 1.
        concurrency = None
        try:
            concurrency = getattr(_cfg, "CARD_SUMMARY_CONCURRENCY", None)
        except Exception:
            concurrency = None
        if concurrency is None:
            try:
                concurrency = int((os.getenv("AIDEV_CARD_SUMMARY_CONCURRENCY") or "").strip() or 5)
            except Exception:
                concurrency = 5
        try:
            concurrency = int(concurrency)
        except Exception:
            concurrency = 5
        if concurrency < 1:
            concurrency = 1

        if compute_embeddings is None:
            compute_embeddings = _env_bool("AIDEV_EMBED_CARDS", False)

        try:
            idx = self._ensure_fresh_index()
            targets = self._list_needs_ai_refresh(idx, paths=paths)

            # Guardrail: never summarize internal/generated artifacts
            targets = [
                t
                for t in targets
                if not _matches_any(_to_posix(t), _INTERNAL_DENY_GLOBS)
            ]

            if max_files is not None and max_files > 0:
                targets = targets[: max_files]

            total_targets = len(targets)

            if not targets:
                return {
                    "ok": True,
                    "total": 0,
                    "updated": 0,
                    "failed": 0,
                    "skipped": 0,
                    "results": [],
                    "message": "No files needed AI summaries.",
                    "error": None,
                }

            results: List[Dict[str, Any]] = []

            # Schedule one per-file summarize call per target concurrently.
            # NOTE: workers MUST NOT mutate the shared idx; we merge deterministically after.
            work_items: List[Tuple[str, Dict[str, Any]]] = []
            for rel in targets:
                meta = idx.get(rel) or {}
                # ensure per-file meta is a dict for safety
                if not isinstance(meta, dict):
                    meta = {}
                work_items.append((rel, meta))

            outcomes: Dict[str, Tuple[Optional[Dict[str, Any]], Dict[str, Any]]] = {}
            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as ex:
                    futs: Dict[concurrent.futures.Future, str] = {}
                    for rel, meta in work_items:
                        fut = ex.submit(
                            self._summarize_one_file,
                            rel,
                            meta,
                            model=model,
                            max_bytes_per_file=max_bytes_per_file,
                            compute_embeddings=compute_embeddings,
                        )
                        futs[fut] = rel

                    for fut in concurrent.futures.as_completed(futs):
                        rel = futs.get(fut) or ""
                        if not rel:
                            continue
                        try:
                            updated_meta, per = fut.result()
                            if not isinstance(per, dict):
                                per = {
                                    "path": rel,
                                    "ok": False,
                                    "summary_len": 0,
                                    "error": "Invalid per-file result.",
                                }
                            outcomes[rel] = (updated_meta, per)
                        except Exception as e:
                            err = str(e)[:400] or e.__class__.__name__
                            logger.warning(
                                "Summarize-changed: worker failed",
                                ctx={"path": rel, "err": err},
                            )
                            outcomes[rel] = (
                                None,
                                {
                                    "path": rel,
                                    "ok": False,
                                    "summary_len": 0,
                                    "error": err,
                                },
                            )
            except Exception as e:
                # If the executor itself fails, fall back to deterministic sequential behavior.
                logger.warning(
                    "Summarize-changed: executor creation/run failed; falling back to sequential",
                    ctx={"err": str(e)[:200]},
                )
                outcomes = {}
                for rel, meta in work_items:
                    try:
                        updated_meta, per = self._summarize_one_file(
                            rel,
                            meta,
                            model=model,
                            max_bytes_per_file=max_bytes_per_file,
                            compute_embeddings=compute_embeddings,
                        )
                        if not isinstance(per, dict):
                            per = {
                                "path": rel,
                                "ok": False,
                                "summary_len": 0,
                                "error": "Invalid per-file result.",
                            }
                        outcomes[rel] = (updated_meta, per)
                    except Exception as ee:
                        err = str(ee)[:400] or ee.__class__.__name__
                        outcomes[rel] = (
                            None,
                            {"path": rel, "ok": False, "summary_len": 0, "error": err},
                        )

            # Deterministic merge + write: iterate targets in order.
            for rel in targets:
                updated_meta, per = outcomes.get(
                    rel,
                    (
                        None,
                        {
                            "path": rel,
                            "ok": False,
                            "summary_len": 0,
                            "error": "Missing worker result.",
                        },
                    ),
                )

                if isinstance(updated_meta, dict):
                    idx[rel] = updated_meta

                    # Persist just this card's JSON file; do not let errors block others.
                    try:
                        self._write_card_file(rel, updated_meta)
                    except Exception as e:
                        werr = str(e)[:200]
                        logger.warning(
                            "Summarize-changed: write_card_file failed",
                            ctx={"path": rel, "err": werr},
                        )
                        # Make the per-file error deterministic and visible.
                        try:
                            per = dict(per)
                            per["ok"] = False
                            per["error"] = (
                                (per.get("error") or "")
                                + ("; " if per.get("error") else "")
                                + f"write_card_file failed: {werr}"
                            )
                        except Exception:
                            per = {
                                "path": rel,
                                "ok": False,
                                "summary_len": 0,
                                "error": f"write_card_file failed: {werr}",
                            }

                # Append report entry in deterministic order.
                try:
                    results.append(
                        {
                            "path": per.get("path", rel),
                            "ok": bool(per.get("ok")),
                            "summary_len": int(per.get("summary_len") or 0),
                            "error": per.get("error"),
                        }
                    )
                except Exception:
                    results.append(
                        {
                            "path": rel,
                            "ok": False,
                            "summary_len": 0,
                            "error": "Failed to record per-file result.",
                        }
                    )

            # Extra visibility for a few key files, including cards.py
            for rel, meta in idx.items():
                if rel in (
                    "aidev/__init__.py",
                    "aidev/api/conversation.py",
                    "aidev/api/llm.py",
                    "aidev/cards.py",
                ):
                    self._ensure_summary_obj(meta)

            updated = sum(1 for r in results if r["ok"])
            failed = sum(1 for r in results if not r["ok"])
            skipped = total_targets - len(results)

            # Single index save at the end
            try:
                self.save_card_index(idx)
            except Exception as se:
                err = str(se)[:400]
                logger.warning(
                    "Summarize-changed: save_card_index failed",
                    ctx={"err": err},
                )
                return {
                    "ok": False,
                    "total": total_targets,
                    "updated": updated,
                    "failed": failed,
                    "skipped": skipped,
                    "results": results,
                    "message": "save_card_index failed: " + err,
                    "error": err,
                }

            message = f"Summarized {updated} file(s), skipped {skipped}, failed {failed}."

            report: Dict[str, Any] = {
                "ok": True,
                "total": total_targets,
                "updated": updated,
                "failed": failed,
                "skipped": skipped,
                "results": results,
                "message": message,
                "error": None,
            }
            return report

        except Exception as e:
            msg = str(e)
            logger.warning("Summarize-changed fatal", ctx={"err": msg[:400]})
            return {
                "ok": False,
                "total": 0,
                "updated": 0,
                "failed": 0,
                "skipped": 0,
                "results": [],
                "message": msg,
                "error": msg,
            }

    def refresh_changed_paths(self, paths: Iterable[str]) -> List[str]:
        """
        Refresh cards for a given iterable of repo-relative paths and return the list
        of paths that were actually refreshed (added/removed/metadata changed).

        Behavior is gated by the REFRESH_CARDS_BETWEEN_RECS configuration boolean (default True).
        Emits optional events if an `aidev.events` module with helper functions is importable.
        This helper is best-effort and never raises; on error it returns an empty list.
        """
        try:
            enabled = getattr(_cfg, "REFRESH_CARDS_BETWEEN_RECS", True)
            if not enabled:
                logger.info("cards refresh between recommendations disabled by config", ctx={})
                return []

            if not paths:
                return []

            # Normalize/deduplicate paths list into repo-relative POSIX strings
            
            provided_raw = list(paths) if not isinstance(paths, str) else [paths]
            provided: List[str] = []
            for p in provided_raw:
                if not p:
                    continue
                try:
                    provided.append(Path(str(p)).as_posix())
                except Exception:
                    # fallback to str(p) with slashes normalized
                    provided.append(str(p).replace("\\", "/"))
            provided = sorted(list(dict.fromkeys([p for p in provided if p])))
            if not provided:
                return []

            # Try to import event emitter (best-effort)
            _events = None
            try:
                from . import events as _events  # type: ignore
            except Exception:
                _events = None

            # session id if available from config
            session_id = getattr(_cfg, "SESSION_ID", None)

            # Emit start event via helper if present, else fall back to generic emit
            start_payload = {"requested_count": len(provided)}
            if session_id is not None:
                start_payload["session_id"] = session_id
            if _events is not None:
                try:
                    if hasattr(_events, "cards_refresh_start"):
                        try:
                            _events.cards_refresh_start(start_payload)
                        except Exception:
                            # helper failure shouldn't stop flow
                            pass
                    elif hasattr(_events, "emit"):
                        try:
                            _events.emit("cards.refresh.start", start_payload)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Snapshot index before refresh for the provided paths
            try:
                idx_before = self.load_card_index() or {}
            except Exception:
                idx_before = {}

            snap_before: Dict[str, Optional[Dict[str, Any]]] = {}
            for p in provided:
                m = idx_before.get(p)
                if isinstance(m, dict):
                    try:
                        s = m.get("summary") or {}
                        ai_sha = None
                        if isinstance(s, dict):
                            ai_sha = s.get("ai_sha")
                        snap_before[p] = {
                            "file_sha": str(m.get("file_sha") or ""),
                            "ai_sha": str(ai_sha or ""),
                            "sha256": str(m.get("sha256") or ""),
                        }
                    except Exception:
                        snap_before[p] = None
                else:
                    snap_before[p] = None

            # Perform an efficient structural update for just these files.
            try:
                # update_cards persists per-file card JSON and index
                self.update_cards(files=provided, changed_only=True, force=False)
            except Exception as e:
                logger.warning("refresh_changed_paths: update_cards failed", ctx={"err": str(e)[:200]})

            # Reload index and compute diffs for the requested paths
            try:
                idx_after = self.load_card_index() or {}
            except Exception:
                idx_after = {}

            changed_paths: List[str] = []
            for p in provided:
                before = snap_before.get(p)
                after_meta = idx_after.get(p)

                # Added (was missing, now present)
                if (before is None) and isinstance(after_meta, dict):
                    changed_paths.append(p)
                    continue
                # Removed (was present, now missing)
                if (before is not None) and not isinstance(after_meta, dict):
                    changed_paths.append(p)
                    continue
                # Both missing -> nothing
                if (before is None) and (after_meta is None):
                    continue

                # Both present -> compare distinguishing fields
                try:
                    after_file_sha = str((after_meta or {}).get("file_sha") or "")
                    after_sha256 = str((after_meta or {}).get("sha256") or "")
                    after_ai_sha = ""
                    try:
                        s_after = (after_meta or {}).get("summary") or {}
                        if isinstance(s_after, dict):
                            after_ai_sha = str(s_after.get("ai_sha") or "")
                    except Exception:
                        after_ai_sha = ""

                    before_file_sha = str((before or {}).get("file_sha") or "")
                    before_sha256 = str((before or {}).get("sha256") or "")
                    before_ai_sha = str((before or {}).get("ai_sha") or "")

                    if after_file_sha != before_file_sha or after_sha256 != before_sha256 or after_ai_sha != before_ai_sha:
                        changed_paths.append(p)
                except Exception:
                    # Be conservative: if comparison fails, mark as changed
                    changed_paths.append(p)

            # Normalize to repo-relative POSIX strings, stable sort & dedupe
            changed_paths = [Path(cp).as_posix() for cp in changed_paths]
            changed_paths = sorted(list(dict.fromkeys(changed_paths)))

            done_payload = {
                "changed_paths": changed_paths,
                "refreshed_count": len(changed_paths),
            }
            if session_id is not None:
                done_payload["session_id"] = session_id

            if _events is not None:
                try:
                    if hasattr(_events, "cards_refresh_done"):
                        try:
                            _events.cards_refresh_done(done_payload)
                        except Exception:
                            pass
                    elif hasattr(_events, "emit"):
                        try:
                            _events.emit("cards.refresh.done", done_payload)
                        except Exception:
                            pass
                except Exception:
                    pass

            return changed_paths

        except Exception as e:
            logger.exception("refresh_changed_paths failed", ctx={"err": str(e)[:400]})
            return []

    # ---------- Project map (unified, owned by KnowledgeBase) ----------
    def build_project_map(
        self,
        *,
        project_meta: dict | None = None,
        include_tree: bool = True,
        include_files: bool = True,
        prefer_ai_summaries: bool = True,
        compact_tree: bool = True,
        max_summary_len: int = 320,
    ) -> Dict[str, Any]:
        """
        Build a single canonical project-map payload (rich variant owned by KnowledgeBase).

        NOTE: the lean, LLM-facing .aidev/project_map.json is owned by aidev.repo_map.
        """
        idx = self.load_card_index()
        if not idx:
            idx = self.update_cards()

        struct_idx = self._load_structure_cache()

        files_payload: List[Dict[str, Any]] = []
        if include_files:
            for rel in sorted(self.structure.keys()):
                ci = idx.get(rel) or {}
                self._ensure_summary_obj(ci)
                si = struct_idx.get(rel) or {}

                s = ci.get("summary") or {}
                if prefer_ai_summaries:
                    summary_text = s.get("ai_text") or s.get("heuristic") or ""
                else:
                    summary_text = s.get("heuristic") or s.get("ai_text") or ""

                files_payload.append(
                    {
                        "path": rel,
                        "kind": ci.get("kind") or si.get("kind") or self.structure.get(rel) or "other",
                        "language": ci.get("language") or _language_for_path(rel),
                        "size": int(ci.get("size") or si.get("size") or 0),
                        "sha256": ci.get("sha256") or si.get("sha256") or "",
                        "summary": summary_text,
                        "chunks": si.get("chunks") or ci.get("chunks") or [],
                        "routes": ci.get("routes") or [],
                        "cli_args": ci.get("cli_args") or [],
                        "env_vars": ci.get("env_vars") or [],
                        "owners": ci.get("owners") or [],
                    }
                )

        tree_payload: Optional[List[Dict[str, Any]]] = None
        if include_tree:
            rel_files = self._km_list_rel_files()
            tree_root: Dict[str, Any] = {"dirs": {}, "files": []}
            for rel in rel_files:
                parts = rel.split("/")
                node = tree_root
                for i, name in enumerate(parts):
                    is_file = i == len(parts) - 1
                    if is_file:
                        node.setdefault("files", []).append(
                            {
                                "name": name,
                                "summary": self._km_summary_for(
                                    rel, max_summary_len=max_summary_len
                                ),
                            }
                        )
                    else:
                        node = node["dirs"].setdefault(
                            name, {"dirs": {}, "files": []}
                        )
            tree_payload = self._km_compactify(tree_root) if compact_tree else tree_root

        return {
            "version": 1,
            "root": str(self.root),
            "generated_at": _now_iso(),
            "meta": dict(project_meta or {}),
            "file_count": len(files_payload) if include_files else 0,
            "files": files_payload if include_files else [],
            "tree": tree_payload if include_tree else None,
        }
    def save_project_map(
        self,
        out_path: str | Path,
        *,
        project_meta: dict | None = None,
        include_tree: bool = True,
        include_files: bool = True,
        prefer_ai_summaries: bool = True,
        compact_tree: bool = True,
        max_summary_len: int = 320,
        pretty: bool = False,
        **compat_kwargs,
    ) -> Path:
        """
        Save a project-map JSON file.

        If the target is .aidev/project_map.json, this delegates to aidev.repo_map.build_project_map
        to ensure the lean schema stays consistent.
        """
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: .aidev/project_map.json is owned by repo_map (lean schema).
        # If a caller tries to write that file via cards.py, delegate to repo_map.
        if outp.name == "project_map.json" and outp.parent.name == ".aidev":
            from .repo_map import build_project_map as _build_lean  # local import

            data = _build_lean(self.root)
        else:
            data = self.build_project_map(
                project_meta=project_meta,
                include_tree=include_tree,
                include_files=include_files,
                prefer_ai_summaries=prefer_ai_summaries,
                compact_tree=compact_tree,
                max_summary_len=max_summary_len,
            )

        tmp = outp.with_suffix(outp.suffix + ".tmp")
        if pretty:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        tmp.replace(outp)
        return outp
    def build_project_map_full(self) -> Dict[str, Any]:
        """
        Convenience wrapper: rich map with file details but no tree.
        """
        return self.build_project_map(include_tree=False, include_files=True)
    # ---------- Card ranking for target selection ----------
    def select_cards(
        self,
        query: str,
        *,
        top_k: int = 20,
        filter_includes: Optional[List[str]] = None,
        filter_excludes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of card dicts ordered by relevance to the query.

        New return shape (was List[Tuple[path, score]]): returns a list of dicts:
            { 'id': <rel>, 'path': <rel>, 'summary': <str>, 'score': <float> }

        Uses AI summary if available, else heuristic summary. Optional glob filters
        narrow the candidate pool. Preserves existing scoring logic; only the
        return shape changed to provide path + summary + score for callers.
        """
        q = (query or "").strip() or "project"
        q_tokens = _tokenize(q)
        q_low = q.lower()

        # Simple intent flags for cheap relevance tweaks
        wants_api = any(kw in q_low for kw in ("api", "endpoint", "rest", "route", "http"))
        wants_cli = any(
            kw in q_low
            for kw in ("cli", "command line", "terminal", "flag", "argument", "arguments")
        )

        idx = self.load_card_index()
        if not idx:
            idx = self.update_cards()

        # Apply optional glob filters to narrow candidates
        if filter_includes or filter_excludes:
            candidates = self.select(filter_includes or [], filter_excludes or [])
        else:
            candidates = sorted(self.structure.keys())

        scored: List[Tuple[str, float]] = []
        for rel in candidates:
            meta = idx.get(rel) or {}
            self._ensure_summary_obj(meta)
            summary = self._effective_summary_text(meta)

            exports = meta.get("exports") or []
            routes = meta.get("routes") or []
            cli_args = meta.get("cli_args") or []
            size = int(meta.get("size") or 0)
            contracts = meta.get("contracts") or {}
            public_api = meta.get("public_api") or []
            if not public_api and isinstance(contracts, dict):
                public_api = contracts.get("public_api") or []

            score = _score_rel(rel, summary or "", q_tokens)

            # Change-awareness boost (v2: via staleness.changed)
            st = meta.get("staleness") or {}
            if st.get("changed"):
                score += 0.8

            # Structural signals
            if exports:
                score += 0.3 * min(5, len(exports))
            if routes:
                score += 0.5 * min(4, len(routes))
            if public_api:
                score += 0.2 * min(8, len(public_api))

            # Query-intent-specific bumps
            if wants_cli and cli_args:
                score += 0.8
            if wants_api and (routes or public_api):
                score += 0.8

            # Very large files are often generic / generated; gently de-emphasize
            if size > 120_000:
                score -= 0.5
            if size > 250_000:
                score -= 0.7

            scored.append((rel, float(score)))

        # Sort descending score, stable by path
        scored.sort(key=lambda x: (-x[1], x[0]))

        # Defensive guards & top_k handling
        try:
            k = int(top_k)
        except Exception:
            k = 20
        if k <= 0:
            return []

        results: List[Dict[str, Any]] = []
        for rel, score in scored[:k]:
            try:
                meta = idx.get(rel) or {}
                self._ensure_summary_obj(meta)
                summary_text = self._effective_summary_text(meta) or ""
            except Exception:
                summary_text = ""
            results.append({
                "id": rel,
                "path": rel,
                "summary": summary_text,
                "score": float(score),
            })

        return results
    # ---------- Minimal map helpers ----------
    def _km_list_rel_files(self) -> List[str]:
        """
        Return a list of known relative file paths, using structure if available,
        falling back to a direct filesystem walk.
        """
        root = Path(self.root)
        files: List[str] = []

        struct = getattr(self, "structure", None)
        if isinstance(struct, dict):
            for rel in sorted(struct.keys()):
                files.append(rel.replace("\\", "/"))

        if files:
            return files

        skip_dirs = {".git", ".aidev", "node_modules", ".venv", "__pycache__"}
        for p in root.rglob("*"):
            if p.is_dir():
                if any(part in skip_dirs for part in p.parts):
                    continue
                continue
            if any(part in skip_dirs for part in p.parts):
                continue
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                continue
            files.append(rel)
        return files

    def _km_compactify(self, node: dict) -> List[dict]:
        out: List[dict] = []
        for dname in sorted(node.get("dirs", {}).keys()):
            child = node["dirs"][dname]
            out.append({"d": dname, "c": self._km_compactify(child)})
        files = sorted(node.get("files", []), key=lambda x: x.get("name", ""))
        for f in files:
            out.append({"f": f["name"], "s": f.get("summary", "")})
        return out

    def _km_summary_for(self, rel: str, max_summary_len: int = 320) -> str:
        """
        Build a concise summary for a given relative path using any available
        knowledge map / AI summaries. Never raises; always returns a string.
        """
        try:
            meta: Any = None

            # 1) If a knowledge-map dict is attached (legacy), prefer it
            try:
                km = getattr(self, "km", None)
                if isinstance(km, dict):
                    meta = km.get(rel)
            except Exception:
                meta = None

            # 2) Fall back to card index if no km entry
            if meta is None:
                try:
                    idx = self.load_card_index()
                    meta = idx.get(rel)
                except Exception:
                    meta = None

            text = self._effective_summary_text(meta)

            if not text:
                return ""

            if max_summary_len and len(text) > max_summary_len:
                text = text[: max(0, max_summary_len - 1)].rstrip() + "…"
            return text
        except Exception:
            return ""

    def _km_fallback_summary_from_file(
        self,
        rel_path: str,
        *,
        max_lines: int = 6,
        max_chars: int = 220,
    ) -> str:
        """
        Legacy helper for directly deriving a summary from file contents.

        Retained for compatibility with any external callers.
        """
        p = Path(self.root) / rel_path
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        out_lines: List[str] = []
        for ln in text.splitlines():
            t = ln.strip()
            if not t:
                continue
            if t.startswith(("#", "//", "/*", "*", ";", "<!--")):
                continue
            out_lines.append(t)
            if len(out_lines) >= max_lines:
                break
        s = " ".join(out_lines)
        return s[:max_chars]


# ----------------- Internals: summarization / scoring -----------------

_FIRST_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+)$", re.M)
_DEF_CLASS_FUNC = re.compile(
    r"^\s*(def|class|function|export\s+(?:default\s+)?function)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
_IMPORT_LINE = re.compile(r"^\s*(from\s+\S+\s+import\s+\S+|import\s+\S+)", re.M)
_XML_TITLE = re.compile(r"<title>(.*?)</title>", re.I | re.S)


def _summarize_rel(rel: str, text: str, *, kind: str) -> str:
    """
    Legacy heuristic summary helper; currently unused by KnowledgeBase but
    retained for potential external callers.
    """
    low = rel.lower()
    if low.endswith((".md", ".mdx", ".rst", ".adoc", ".txt")):
        m = _FIRST_HEADING.search(text)
        if m:
            return f"Doc: {m.group(2).strip()}"
        return f"Doc: {rel}"
    if low.endswith((".html", ".xml", ".xhtml")):
        m = _XML_TITLE.search(text)
        if m:
            return f"HTML/XML: {m.group(1).strip()}"
        m2 = _FIRST_HEADING.search(text)
        if m2:
            return f"HTML/XML: {m2.group(2).strip()}"
        return f"HTML/XML: {rel}"
    if low.endswith((".json", ".toml", ".ini", ".cfg", ".yml", ".yaml")) or kind in {
        "build-config",
        "config",
    }:
        keys = _top_level_keys(text, max_keys=6)
        if keys:
            return f"Config: {', '.join(keys)}"
        return f"Config: {rel}"
    if kind == "styles" or low.endswith((".css", ".scss", ".sass", ".less")):
        return f"Stylesheet: {rel}"
    defs = _DEF_CLASS_FUNC.findall(text)
    if defs:
        names = [n for _, n in defs[:6]]
        return f"Source: {', '.join(names)}"
    imports = _IMPORT_LINE.findall(text)
    if imports:
        mods = [
            i.replace("import", "").replace("from", "").strip().split()[0]
            for i in imports[:6]
        ]
        return f"Source imports: {', '.join(mods)}"
    return f"{kind.capitalize()}: {rel}"


def _top_level_keys(text: str, max_keys: int = 6) -> List[str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [str(k) for k in list(obj.keys())[:max_keys]]
    except Exception:
        pass
    keys: List[str] = []
    for line in text.splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            k = line.split(":", 1)[0].strip()
            if k and k not in keys:
                keys.append(k)
            if len(keys) >= max_keys:
                break
    return keys


_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s)]


def _score_rel(rel: str, summary: str, q_tokens: List[str]) -> float:
    rel_low = rel.lower()
    path_tokens = _tokenize(rel_low)
    summ_tokens = _tokenize(summary)
    score = 0.0
    for t in q_tokens:
        score += 3.0 * path_tokens.count(t)
        score += 1.5 * summ_tokens.count(t)
    phrase = " ".join(q_tokens[:3])
    if phrase and phrase in rel_low:
        score += 2.0
    return score


# ----------------- Module-level forwarders -----------------


def summarize_changed(kb: KnowledgeBase, **kwargs) -> Dict[str, Any]:
    return kb.summarize_changed(**kwargs)