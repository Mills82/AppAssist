# aidev/structure.py
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple, Optional

# ---------- Workspace discovery ----------

_PROJECT_MARKERS = {
    # JS/TS/Web
    "package.json": 6,
    "pnpm-lock.yaml": 2,
    "yarn.lock": 2,
    "vite.config.*": 2,
    "next.config.*": 3,
    "nuxt.config.*": 3,
    "angular.json": 3,
    "tsconfig.json": 2,
    # Python
    "pyproject.toml": 6,
    "requirements.txt": 4,
    "Pipfile": 3,
    "poetry.lock": 3,
    "setup.cfg": 2,
    "manage.py": 4,
    # Dart/Flutter
    "pubspec.yaml": 6,
    ".metadata": 1,
    # Mobile / others
    "app.json": 2,             # Expo/React Native
    "AndroidManifest.xml": 2,
    "build.gradle": 2,
    "Cargo.toml": 5,           # Rust
    "go.mod": 5,               # Go
    "composer.json": 4,        # PHP
    "*.sln": 4,
    "*.csproj": 3,             # .NET
}

# Keep discovery snappy by skipping only well-known heavy/cached/build dirs.
# (You can still "include" anything explicitly via .aidev.json includes.)
_EARLY_SKIP_DIRS = {
    # our internal and VCS
    ".aidev", ".git", ".hg", ".svn",
    # JS/web tooling
    "node_modules", "bower_components", ".next", ".nuxt", ".svelte-kit", ".turbo",
    "dist", "build", "out", ".output",
    # language eco systems
    ".gradle", "target", ".dart_tool",
    # python
    ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".cache",
    # IDEs
    ".idea", ".vscode",
    # Apple/iOS build caches
    "Pods", "DerivedData",
    # coverage/artifacts
    "coverage", "htmlcov",
}


@dataclass
class ProjectCandidate:
    path: Path
    score: int
    markers: Dict[str, int]
    language_kinds: List[str]


def _score_dir_for_project(d: Path) -> Tuple[int, Dict[str, int]]:
    hits: Dict[str, int] = {}
    score = 0
    try:
        names = set(os.listdir(d))
    except Exception:
        return 0, {}

    def _match(glob_pat: str) -> bool:
        if "*" in glob_pat or "?" in glob_pat:
            return any(fnmatch(n, glob_pat) for n in names)
        return glob_pat in names

    for pat, w in _PROJECT_MARKERS.items():
        if _match(pat):
            hits[pat] = w
            score += w

    # Heuristic bonus: presence of typical source folders/files
    for folder, w in (("src", 2), ("lib", 1), ("app", 1)):
        if folder in names and (d / folder).is_dir():
            score += w
            hits[f"{folder}/"] = hits.get(f"{folder}/", 0) + w

    return score, hits


def find_projects(
    workspace_root: Path,
    *,
    max_depth: int = 3,
    limit: int = 20,
) -> List[ProjectCandidate]:
    """
    Breadth-first search for likely project roots under `workspace_root`, scored by markers.
    """
    root = Path(workspace_root).resolve()
    if not root.exists():
        return []

    cands: List[ProjectCandidate] = []

    # Always consider the root itself
    s, m = _score_dir_for_project(root)
    if s:
        cands.append(ProjectCandidate(path=root, score=s, markers=m, language_kinds=[]))

    # BFS up to max_depth
    for dirpath, dirnames, _filenames in os.walk(root):
        rel_depth = len(Path(dirpath).relative_to(root).parts)

        # prune early (prevents descending into heavy dirs)
        dirnames[:] = sorted(d for d in dirnames if d not in _EARLY_SKIP_DIRS)
        if rel_depth > max_depth:
            dirnames[:] = []
            continue

        d = Path(dirpath)
        s, m = _score_dir_for_project(d)
        if s:
            cands.append(ProjectCandidate(path=d, score=s, markers=m, language_kinds=[]))

    # Detect language kinds for each
    for c in cands:
        c.language_kinds = sorted(_detect_language_kinds_in_dir(c.path))

    # Sort by score desc, then shorter depth, then name
    cands.sort(key=lambda x: (-x.score, len(x.path.parts), x.path.name.lower()))
    return cands[:limit]


# ---------- Tree walk / include-exclude handling ----------

def _matches_any(rel: str, patterns: List[str]) -> bool:
    """Shell-style matching across path separators."""
    if not patterns:
        return True
    rel = rel.replace("\\", "/")
    # Path.match doesn't support recursive **; fnmatch treats '/' as ordinary char,
    # which gives the inclusive behavior people expect in include/exclude globs.
    return any(fnmatch(rel, pat) for pat in patterns)


def _load_globs_from_config(root: Path, includes: Optional[List[str]], excludes: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """
    If the caller didn’t specify includes/excludes, fall back to .aidev.json (if present).
    """
    # If either is explicitly provided (not None), normalize and return them.
    if includes is not None or excludes is not None:
        return includes or [], excludes or []

    cfg_path = Path(root) / ".aidev.json"
    try:
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                inc = cfg.get("include") or []
                exc = cfg.get("exclude") or []
                if isinstance(inc, list):
                    includes = [str(x) for x in inc if isinstance(x, str)]
                if isinstance(exc, list):
                    excludes = [str(x) for x in exc if isinstance(x, str)]
    except Exception:
        # best-effort only; silently ignore config parse errors
        pass

    return includes or [], excludes or []


def iter_files(root: Path, includes: Optional[List[str]] = None, excludes: Optional[List[str]] = None) -> Iterable[Path]:
    """
    Walk the tree under `root`, applying include/exclude globs and early directory pruning.
    Yields absolute Paths in a deterministic order.
    """
    root = Path(root)
    early_skip = _EARLY_SKIP_DIRS

    # Normalize None -> [] for internal use
    includes = includes or []
    excludes = excludes or []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune traversal into skip dirs and keep ordering deterministic
        dirnames[:] = sorted(d for d in dirnames if d not in early_skip)

        # if this path is inside a skip dir, skip its files (kept for safety)
        rel_parts = Path(dirpath).relative_to(root).parts
        if any(part in early_skip for part in rel_parts):
            continue

        for name in sorted(filenames):
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()

            # hard guard for .aidev
            if rel.split("/", 1)[0] == ".aidev":
                continue

            # apply excludes first
            if excludes and _matches_any(rel, excludes):
                continue

            # then includes (if provided)
            if includes and not _matches_any(rel, includes):
                continue

            yield p


# ---------- Comment stripping (prompt-only) ----------

_comment_re_js = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_comment_re_hash = re.compile(r"#[^\n]*")
_comment_re_xml = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_comments_for_ext(s: str, rel: str) -> str:
    low = rel.lower()
    try:
        if low.endswith((
            ".js", ".ts", ".jsx", ".tsx", ".dart",
            ".java", ".kt", ".kts", ".c", ".cpp",
            ".cs", ".swift", ".go", ".php", ".rs",
        )):
            s = _comment_re_js.sub("", s)

        if low.endswith((
            ".yml", ".yaml", ".sh", ".py", ".toml", ".ini",
            ".cfg", ".conf", ".rb", ".pl", ".r", ".tex",
            ".mk", ".makefile",
        )) or "/.github/" in rel:
            s = _comment_re_hash.sub("", s)

        if low.endswith((".xml", ".html", ".xhtml", ".vue")):
            s = _comment_re_xml.sub("", s)
    except Exception:
        # strictly best-effort; don’t block structure discovery on comment stripping
        pass
    return s


# ---------- Chunking & hashing ----------

MAX_FILE_BYTES = 1_000_000  # read cap per file for sampling (1 MB); full hashing streams entire file


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def _stable_chunk_indices(
    text: str,
    target_bytes: int = 8_000,
    slack_bytes: int = 2_000,
) -> List[Tuple[int, int]]:
    """
    Content-aware, deterministic chunking:
    - Aim for ~target_bytes per chunk.
    - Choose boundaries near blank lines or definition headers within +/- slack.
    - Falls back to hard split if nothing found.
    Returns list of (start, end) indices in character offsets.
    """
    if not text:
        return []

    b = text.encode("utf-8", errors="ignore")
    n = len(b)
    if n <= target_bytes + 512:
        return [(0, len(text))]

    # Precompute candidate split points by scanning text
    # Prefer: blank lines, def/class/function headers, closing braces, import blocks
    lines = text.splitlines(keepends=True)
    offsets: List[int] = []
    pos = 0
    header_pat = re.compile(
        r"^\s*(def |class |function |export |public |private |struct |interface |\}|# )",
        re.I,
    )
    for ln in lines:
        if ln.strip() == "" or header_pat.search(ln):
            offsets.append(pos)
        pos += len(ln)
    offsets.append(len(text))

    # Helper: find nearest offset (by char index) to desired byte position
    def nearest_offset(target_b: int) -> int:
        approx_char = int(target_b / max(1, n) * len(text))
        lo = max(0, approx_char - 2000)
        hi = min(len(text), approx_char + 2000)
        candidates = [o for o in offsets if lo <= o <= hi]
        if not candidates:
            candidates = offsets
        # Measure error in bytes rather than chars
        return min(
            candidates,
            key=lambda o: abs(len(text[:o].encode("utf-8", "ignore")) - target_b),
        )

    chunks: List[Tuple[int, int]] = []
    byte_cursor = 0
    char_cursor = 0
    while byte_cursor < n:
        target = byte_cursor + target_bytes
        min_b = byte_cursor + max(2_000, target_bytes - slack_bytes)
        max_b = min(n, byte_cursor + target_bytes + slack_bytes)
        if min_b >= n:
            chunks.append((char_cursor, len(text)))
            break

        bound_b = min(max(nearest_offset(target), 0), len(text))
        bound_b_as_bytes = len(text[:bound_b].encode("utf-8", "ignore"))
        if bound_b_as_bytes < min_b or bound_b_as_bytes > max_b:
            # fall back to a simple byte window if we couldn't find a good semantic boundary
            bound_chars = len(b[:max_b].decode("utf-8", "ignore"))
            end_char = max(bound_chars, char_cursor + 1)
        else:
            end_char = max(bound_b, char_cursor + 1)

        chunks.append((char_cursor, end_char))
        byte_cursor = len(text[:end_char].encode("utf-8", "ignore"))
        char_cursor = end_char

    return chunks


def _file_chunks_with_hashes(path: Path, rel: str, body: str) -> List[Dict[str, object]]:
    chunks_idx = _stable_chunk_indices(body)
    out: List[Dict[str, object]] = []
    for (a, b) in chunks_idx:
        chunk_text = body[a:b]
        h = hashlib.sha256(chunk_text.encode("utf-8", errors="ignore")).hexdigest()
        out.append(
            {
                "rel": rel,
                "start": a,
                "end": b,
                "sha256": h,
                "bytes": len(chunk_text.encode("utf-8", "ignore")),
            }
        )
    return out


# ---------- Cache (hash-based incremental) ----------

def _cache_path(root: Path) -> Path:
    return root / ".aidev" / "cache" / "file_index.json"


def _load_index_cache(root: Path) -> Dict[str, Dict[str, object]]:
    p = _cache_path(root)
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_index_cache(root: Path, index: Dict[str, Dict[str, object]]) -> None:
    try:
        p = _cache_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        # best-effort only
        pass


# ---------- Structure & context sampling ----------

def discover_structure(
    root: Path,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    max_total_kb: int = 1024,
    strip_comments: bool = False,
) -> Tuple[Dict[str, str], str]:
    """
    Return (structure_map, truncated_context_blob).

    structure_map: rel_path -> coarse kind ("source","styles","config","docs","assets","other")
    Context blob is assembled from stable file chunks up to the KB budget.

    Notes:
    - We always index *all* files that pass include/exclude, even after the
      context budget is hit. Only the context blob stops growing.
    - Includes/excludes default to .aidev.json if both lists are empty.
    """
    root = Path(root)

    # If caller didn't pass filters, fall back to .aidev.json (if present)
    includes, excludes = _load_globs_from_config(root, includes, excludes)

    index_cache = _load_index_cache(root)
    struct: Dict[str, str] = {}
    total_kb = 0
    lines: List[str] = []

    budget_hit = False

    for p in iter_files(root, includes, excludes):
        rel = p.relative_to(root).as_posix()
        kind = _kind_for_path(rel)
        struct[rel] = kind

        # Incremental hash reuse
        try:
            stat = p.stat()
        except Exception:
            continue

        cached = index_cache.get(rel) or {}
        same_mtime = cached.get("mtime") == getattr(stat, "st_mtime", None)
        same_size = cached.get("size") == getattr(stat, "st_size", None)

        if same_mtime and same_size:
            sha = cached.get("sha256", "")
            chunks = cached.get("chunks", [])
            body = None  # defer reading unless needed for context
        else:
            # Read body (capped for sampling but hash the full file)
            try:
                # Full hash first (streaming)
                sha = _sha256_file(p)
                with open(p, "rb") as f:
                    raw = f.read(MAX_FILE_BYTES)
                    _ = f.read(1)  # sentinel read to detect truncation, intentionally unused
                body = raw.decode("utf-8", errors="replace")
                # optional comment strip (prompt-only)
                if strip_comments:
                    body = _strip_comments_for_ext(body, rel)
                chunks = _file_chunks_with_hashes(p, rel, body)
            except Exception:
                sha = ""
                chunks = []
                body = ""

            # Update cache entry
            index_cache[rel] = {
                "sha256": sha,
                "mtime": getattr(stat, "st_mtime", None),
                "size": getattr(stat, "st_size", None),
                "kind": kind,
                "chunks": chunks,
            }

        # Assemble context, but do NOT stop indexing when budget is reached
        if total_kb >= max_total_kb:
            continue

        # If we didn't read body but need text for context, read lazily only until budget
        if cached and same_mtime and same_size and body is None:
            try:
                with open(p, "rb") as f:
                    raw = f.read(MAX_FILE_BYTES)
                    _ = f.read(1)
                body = raw.decode("utf-8", errors="replace")
                if strip_comments:
                    body = _strip_comments_for_ext(body, rel)
            except Exception:
                body = ""

        # choose as many leading chunks as fit in remaining budget
        ch_list = chunks if isinstance(chunks, list) else []
        for ch in ch_list:
            if total_kb >= max_total_kb:
                break

            start = int(ch.get("start", 0))
            end = int(ch.get("end", 0))
            bytes_len = int(ch.get("bytes", 0))

            size_kb = max(1, (bytes_len // 1024) + 1)
            if total_kb + size_kb > max_total_kb:
                if not budget_hit:
                    lines.append("// [context budget reached; remaining files omitted]\n")
                    budget_hit = True
                total_kb = max_total_kb
                break

            lines.append(f"// {rel}  sha256:{sha}  chunk:{start}-{end}\n")

            if body is not None:
                lines.append(body[start:end])
            else:
                # Extremely rare path: chunks from cache, body missing, and we
                # haven't already read it lazily above.
                try:
                    with open(p, "rb") as f:
                        raw = f.read(MAX_FILE_BYTES)
                        _ = f.read(1)
                    text2 = raw.decode("utf-8", errors="replace")
                    if strip_comments:
                        text2 = _strip_comments_for_ext(text2, rel)
                    lines.append(text2[start:end])
                except Exception:
                    # best-effort only
                    pass

            total_kb += size_kb

    # Persist cache (best-effort)
    _save_index_cache(root, index_cache)

    return struct, "".join(lines)


def get_structure_slices(
    root: Path,
    targets: Optional[List[str]] = None,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    max_total_bytes: int = 200_000,
    max_items: int = 1000,
    strip_comments: bool = False,
) -> List[Dict[str, object]]:
    """
    Return a deterministic, ordered list of structure "slices" (file chunks) suitable
    for evidence gathering.

    Each entry includes only repo-relative provenance fields:
      {"rel","start","end","bytes","sha256","file_sha256","ordinal"}

    Determinism guarantees:
    - File ordering is lexicographic by repo-relative path.
    - Chunk ordering is by ascending start offset.
    - If truncation occurs, a final marker entry is appended:
      {"truncated": true, "reason": "budget", "ordinal": N}

    Examples:
      slices = get_structure_slices(Path("."), targets=["src/main.py"], max_items=50)
      slices = get_structure_slices(Path("."), includes=["src/**"], excludes=["**/node_modules/**"])
    """
    root = Path(root)
    root_resolved = root.resolve()

    targets = targets or []

    # If caller didn't pass filters, fall back to .aidev.json (if present)
    includes, excludes = _load_globs_from_config(root, includes, excludes)

    index_cache = _load_index_cache(root)

    # Build structure map by walking files under root, without producing context.
    struct: Dict[str, str] = {}
    for p in iter_files(root, includes, excludes):
        try:
            rel = p.relative_to(root).as_posix()
        except Exception:
            # hard guarantee: never emit absolute paths; if rel can't be computed, skip
            continue
        struct[rel] = _kind_for_path(rel)

    # Filter to a relevant subtree if targets were provided
    struct2 = subtree_structure(struct, targets)

    rels = sorted(struct2.keys())

    out: List[Dict[str, object]] = []
    total_bytes = 0
    ordinal = 0
    truncated = False
    cache_changed = False

    for rel in rels:
        # Ensure path stays within root and never follow outside
        try:
            abs_path = (root / rel)
            # strict check: resolve and ensure still within root
            abs_resolved = abs_path.resolve()
            abs_resolved.relative_to(root_resolved)
        except Exception:
            continue

        try:
            stat = abs_path.stat()
        except Exception:
            continue

        cached = index_cache.get(rel) or {}
        same_mtime = cached.get("mtime") == getattr(stat, "st_mtime", None)
        same_size = cached.get("size") == getattr(stat, "st_size", None)

        file_sha = ""
        chunks: List[Dict[str, object]] = []

        if same_mtime and same_size and isinstance(cached.get("chunks"), list):
            file_sha = str(cached.get("sha256", "") or "")
            chunks = cached.get("chunks", []) or []
        else:
            # deterministically compute and cache chunks
            try:
                file_sha = _sha256_file(abs_path)
                with open(abs_path, "rb") as f:
                    raw = f.read(MAX_FILE_BYTES)
                    _ = f.read(1)
                body = raw.decode("utf-8", errors="replace")
                if strip_comments:
                    body = _strip_comments_for_ext(body, rel)
                chunks = _file_chunks_with_hashes(abs_path, rel, body)
            except Exception:
                file_sha = ""
                chunks = []

            index_cache[rel] = {
                "sha256": file_sha,
                "mtime": getattr(stat, "st_mtime", None),
                "size": getattr(stat, "st_size", None),
                "kind": cached.get("kind") or struct.get(rel) or _kind_for_path(rel),
                "chunks": chunks,
            }
            cache_changed = True

        # Ensure deterministic ordering of chunks
        if not isinstance(chunks, list):
            continue
        chunks_sorted = sorted(chunks, key=lambda ch: int((ch or {}).get("start", 0)))

        for ch in chunks_sorted:
            if ordinal >= max_items:
                truncated = True
                break

            start = int((ch or {}).get("start", 0))
            end = int((ch or {}).get("end", 0))
            bytes_len = int((ch or {}).get("bytes", 0))
            chunk_sha = str((ch or {}).get("sha256", "") or "")

            # Enforce total byte budget
            if max_total_bytes is not None and total_bytes + max(0, bytes_len) > max_total_bytes:
                truncated = True
                break

            out.append(
                {
                    "rel": rel,
                    "start": start,
                    "end": end,
                    "bytes": bytes_len,
                    "sha256": chunk_sha,
                    "file_sha256": file_sha,
                    "ordinal": ordinal,
                }
            )
            ordinal += 1
            total_bytes += max(0, bytes_len)

        if truncated:
            break

    if truncated:
        out.append({"truncated": True, "reason": "budget", "ordinal": ordinal})

    if cache_changed:
        _save_index_cache(root, index_cache)

    return out


def _kind_for_path(rel: str) -> str:
    low = rel.lower()
    if low.endswith((
        ".ts", ".tsx", ".js", ".jsx", ".dart", ".py",
        ".php", ".kt", ".kts", ".java", ".rs", ".go",
        ".c", ".cpp", ".cs", ".swift",
    )):
        return "source"

    if low.endswith((".css", ".scss", ".sass", ".less", ".pcss")):
        return "styles"

    base = os.path.basename(rel)
    if base in {
        "package.json",
        "tsconfig.json",
        "next.config.js",
        "next.config.mjs",
        "pyproject.toml",
        "requirements.txt",
        "composer.json",
    }:
        return "build-config"

    if low.endswith((".json", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".xml")):
        return "config"

    if low.endswith((".md", ".txt", ".rst", ".adoc", ".org", ".mdx")):
        return "docs"

    if rel.startswith("public/") or low.endswith((
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    )):
        return "assets"

    return "other"


def compact_structure(struct: Dict[str, str]) -> Dict[str, object]:
    """
    Produce a compact index for prompts: total count, by top dir, by extension,
    plus language kinds.

    This is intentionally light-weight; detailed per-file information lives in
    KnowledgeBase.save_project_map() and is attached by the orchestrator as
    meta["project_map"].
    """
    by_top: DefaultDict[str, int] = defaultdict(int)
    by_ext: DefaultDict[str, int] = defaultdict(int)

    for rel in struct.keys():
        top = rel.split("/", 1)[0]
        by_top[top] += 1
        ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else "(noext)"
        by_ext[ext] += 1

    language_kinds = sorted(_detect_language_kinds_from_struct(struct))
    return {
        "total_files": len(struct),
        "by_top": dict(sorted(by_top.items())),
        "by_ext": dict(sorted(by_ext.items())),
        "language_kinds": language_kinds,
    }


def subtree_structure(struct: Dict[str, str], targets: List[str]) -> Dict[str, str]:
    """
    Filter structure to entries that share a top-level dir or direct parent with any target.
    """
    if not targets:
        return struct

    tops: Set[str] = set()
    parents: Set[str] = set()

    for t in targets:
        parts = t.split("/")
        if parts:
            tops.add(parts[0])
        if len(parts) > 1:
            parents.add("/".join(parts[:-1]))

    out: Dict[str, str] = {}
    for rel, kind in struct.items():
        if rel.split("/", 1)[0] in tops:
            out[rel] = kind
            continue
        parent = "/".join(rel.split("/")[:-1])
        if parent in parents:
            out[rel] = kind
            continue

    return out


# ---------- Language detection ----------

_LANG_BY_EXT = {
    ".py": "python",
    ".ipynb": "python",
    ".js": "node",
    ".jsx": "node",
    ".ts": "node",
    ".tsx": "node",
    ".dart": "dart",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".cs": "csharp",
    ".c": "c",
    ".cpp": "cpp",
    ".m": "objc",
    ".mm": "objc++",
}


def _detect_language_kinds_in_dir(d: Path) -> Set[str]:
    kinds: Set[str] = set()
    try:
        names = set(os.listdir(d))
    except Exception:
        names = set()

    def has(p: str) -> bool:
        if "*" in p or "?" in p:
            return any(fnmatch(n, p) for n in names)
        return p in names

    if has("pubspec.yaml"):
        kinds.add("dart")
        kinds.add("flutter")

    if has("package.json"):
        kinds.add("node")
        for cfg in (
            "next.config.js",
            "next.config.mjs",
            "nuxt.config.ts",
            "nuxt.config.js",
            "angular.json",
        ):
            if has(cfg):
                kinds.add("web")

    if has("pyproject.toml") or has("requirements.txt"):
        kinds.add("python")

    if has("Cargo.toml"):
        kinds.add("rust")

    if has("go.mod"):
        kinds.add("go")

    if has("composer.json"):
        kinds.add("php")

    if any(has(p) for p in ("*.sln", "*.csproj")):
        kinds.add("csharp")

    try:
        for n in names:
            ext = "." + n.rsplit(".", 1)[-1] if "." in n else ""
            if ext in _LANG_BY_EXT:
                kinds.add(_LANG_BY_EXT[ext])
    except Exception:
        pass

    return kinds


def _detect_language_kinds_from_struct(struct: Dict[str, str]) -> Set[str]:
    kinds: Set[str] = set()
    for rel in struct.keys():
        ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else ""
        if ext in _LANG_BY_EXT:
            kinds.add(_LANG_BY_EXT[ext])
        if rel.endswith("pubspec.yaml"):
            kinds.add("dart")
            kinds.add("flutter")
        if rel.endswith("package.json"):
            kinds.add("node")
    return kinds
