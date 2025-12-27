# aidev/context/edit_context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..cards import KnowledgeBase, _language_for_path  # type: ignore[attr-defined]

"""
Edit-context helpers.

This module builds a small "context bubble" around a target file before asking
the LLM to generate edits. It pulls in:

- The primary file being edited.
- Local neighbors in the same directory.
- Direct dependencies (what the file imports).
- Direct dependents (who imports this file).
- Associated tests/configs, when we have hints.

The main entrypoints are:

    build_context_for_edit(target_path, kb_or_map, ...)
    build_context_bundle_for_paths(kb, project_root, paths, ...)

`kb_or_map` can be either:
- a `KnowledgeBase` instance (preferred), or
- a project map dict as produced by `aidev.repo_map.build_project_map(...)`
  or `KnowledgeBase.build_project_map_full()`.

This module is intentionally side-effect free: it never mutates cards, indices,
or any on-disk structures.
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContextFile:
    """
    A single file included in the edit/analyze context bundle.

    Attributes:
        path:     Project-relative path (POSIX style, e.g. "src/app.tsx").
        role:     Rough role in the context bubble:
                  "primary", "same_dir", "dependency", "dependent",
                  "test", "config", or "neighbor".
        content:  Text content (truncated to `max_bytes_per_file`).
        summary:  Short description (AI/heuristic/empty); safe to show to LLM.
        language: Approximate language (see `_language_for_path`).
        kind:     Structural kind if we know it (from `structure`/project map).
        size:     File size in bytes (best-effort).
        reason:   Optional human-readable explanation of *why* this file
                  is included in the context.
        weight:   Numeric hint for ordering/importance inside the context.
    """

    path: str
    role: str
    content: str
    summary: str
    language: str
    kind: Optional[str]
    size: int
    reason: Optional[str] = None
    weight: float = 0.0


# A "bundle" is the full ordered context bubble around a single primary file.
ContextBundle = List[ContextFile]

KbOrMap = Union[KnowledgeBase, Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _normalize_rel(root: Path, target_path: str) -> str:
    """
    Turn an arbitrary path (abs/rel/mixed) into a project-relative POSIX path.
    """
    p = Path(target_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        # If the path is outside the project root, just use a POSIXified string.
        rel = _to_posix(target_path)
    return rel


def _read_file_text(
    root: Path,
    rel: str,
    *,
    max_bytes: int,
) -> Tuple[str, int]:
    """
    Best-effort read of a file as UTF-8 text, truncated to `max_bytes`.

    Returns (content, size_bytes).
    """
    p = root / rel
    try:
        with p.open("rb") as f:
            raw = f.read(max_bytes)
            _ = f.read(1)  # sentinel to detect truncation, but we don't use it
        size = p.stat().st_size
        text = raw.decode("utf-8", errors="replace")
        return text, int(size)
    except Exception:
        # If anything goes wrong, treat as empty (but keep size best-effort).
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        return "", int(size)


def _project_root_from_kb_or_map(kb_or_map: KbOrMap) -> Path:
    if isinstance(kb_or_map, KnowledgeBase):
        return Path(kb_or_map.root).resolve()
    root_str = str(kb_or_map.get("root") or ".")
    return Path(root_str).resolve()


def _structure_from_kb_or_map(
    kb_or_map: KbOrMap,
) -> Dict[str, str]:
    """
    Return a mapping of rel_path -> kind, best-effort.

    If we only have a project map, we synthesize a minimal structure from it.
    """
    if isinstance(kb_or_map, KnowledgeBase):
        return dict(kb_or_map.structure)

    # Project map dict variant: expect `files` with { path, kind, ... } entries.
    struct: Dict[str, str] = {}
    files = kb_or_map.get("files") or []
    if isinstance(files, Sequence):
        for rec in files:
            if not isinstance(rec, Mapping):
                continue
            path = rec.get("path")
            if not isinstance(path, str):
                continue
            kind = rec.get("kind") or "other"
            struct[_to_posix(path)] = str(kind)
    return struct


def _summary_from_kb(
    kb: KnowledgeBase,
    rel: str,
    *,
    max_len: int = 320,
) -> str:
    """
    Use KnowledgeBase to obtain a summary for a file, best-effort.
    """
    # Prefer the existing km/card machinery if available.
    try:
        if hasattr(kb, "_km_summary_for"):
            text = kb._km_summary_for(rel, max_summary_len=max_len)  # type: ignore[attr-defined]
            if text:
                return text
    except Exception:
        pass

    # Fallback: card index entries
    try:
        idx = kb.load_card_index()
        meta = idx.get(rel) or {}
        if isinstance(meta, dict):
            kb._ensure_summary_obj(meta)
            s = meta.get("summary") or {}
            text = (
                s.get("ai_text")
                or s.get("heuristic")
                or meta.get("ai_summary")
                or ""
            )
            text = " ".join(str(text).split())
            if max_len and len(text) > max_len:
                text = text[: max(0, max_len - 1)].rstrip() + "…"
            return text
    except Exception:
        pass

    return ""


def _summary_from_project_map(
    project_map: Mapping[str, Any],
    rel: str,
    *,
    max_len: int = 320,
) -> str:
    try:
        files = project_map.get("files") or []
        for rec in files:
            if not isinstance(rec, Mapping):
                continue
            if _to_posix(str(rec.get("path"))) == _to_posix(rel):
                text = rec.get("summary") or ""
                text = " ".join(str(text).split())
                if max_len and len(text) > max_len:
                    text = text[: max(0, max_len - 1)].rstrip() + "…"
                return text
    except Exception:
        pass
    return ""


def _get_related_from_kb(
    kb: KnowledgeBase,
    rel: str,
) -> Dict[str, List[str]]:
    """
    Prefer KnowledgeBase.get_related_files if present, else fall back
    to a simpler heuristic that only considers same-dir siblings.
    """
    if hasattr(kb, "get_related_files"):
        try:
            out = kb.get_related_files(rel)  # type: ignore[call-arg]
            if isinstance(out, Mapping):
                # Normalize structure
                return {
                    "same_dir": list(out.get("same_dir") or []),
                    "dependencies": list(out.get("dependencies") or []),
                    "dependents": list(out.get("dependents") or []),
                    "tests": list(out.get("tests") or []),
                }
        except Exception:
            # Fall back to heuristic below
            pass

    # Heuristic fallback: only same-directory siblings.
    same_dir: List[str] = []
    try:
        target_parent = Path(rel).parent
        for other in kb.structure.keys():
            if other == rel:
                continue
            if Path(other).parent == target_parent:
                same_dir.append(other)
    except Exception:
        same_dir = []

    same_dir = sorted(dict.fromkeys(same_dir))
    return {
        "same_dir": same_dir,
        "dependencies": [],
        "dependents": [],
        "tests": [],
    }


def _role_weight(role: str) -> float:
    """
    Coarse importance ordering for roles; higher is more important.
    """
    table = {
        "primary": 100.0,
        "test": 40.0,
        "dependency": 35.0,
        "dependent": 30.0,
        "config": 25.0,
        "same_dir": 20.0,
        "neighbor": 10.0,
    }
    return table.get(role, 0.0)


def extract_neighbor_snippets(
    bundle: ContextBundle,
    max_snippet_chars: int = 800,
) -> List[Dict[str, str]]:
    """
    Convert a ContextBundle into a compact [{path, snippet}, ...] list for
    neighbors of the primary file.

    Rules:
      - Skip the primary file itself.
      - Prefer `summary` when available.
      - Otherwise, fall back to the first ~max_snippet_chars characters of
        the file content, trying to cut on line boundaries when possible.
    """
    if not bundle:
        return []

    # Assume the first entry is the primary file (by construction of the bundle).
    primary_path = bundle[0].path

    out: List[Dict[str, str]] = []
    for cf in bundle:
        # Skip the primary file itself.
        if cf.path == primary_path:
            continue

        # 1) Prefer the existing summary, if present.
        snippet = (cf.summary or "").strip()

        # 2) Fallback: first N chars of content, ideally on line boundaries.
        if not snippet:
            raw = (cf.content or "").strip()
            if not raw:
                continue

            if max_snippet_chars and len(raw) > max_snippet_chars:
                # Try to accumulate whole lines up to the character budget.
                lines = raw.splitlines()
                acc_lines: List[str] = []
                total = 0
                for ln in lines:
                    # +1 for the newline we’d reinsert when joining
                    ln_len = len(ln) + 1
                    if total + ln_len > max_snippet_chars:
                        break
                    acc_lines.append(ln)
                    total += ln_len

                snippet = "\n".join(acc_lines).strip()
                if not snippet:
                    # Fallback if line-based trimming produced nothing.
                    snippet = raw[:max_snippet_chars].rstrip()
                snippet = snippet + "…"
            else:
                snippet = raw

        if snippet:
            out.append(
                {
                    "path": cf.path,
                    "snippet": snippet,
                }
            )

    return out


# ---------------------------------------------------------------------------
# Public API — single-path bundles
# ---------------------------------------------------------------------------


def build_context_for_edit(
    target_path: str,
    kb_or_map: KbOrMap,
    *,
    max_neighbors: int = 16,
    max_tests: int = 8,
    max_bytes_per_file: int = 80_000,
) -> ContextBundle:
    """
    Build a per-edit context bundle for `target_path`.

    Args:
        target_path:
            The file we intend to edit (abs or rel); will be normalized to a
            project-relative POSIX path.

        kb_or_map:
            Either:
              - a `KnowledgeBase` instance (preferred, richer graph info), or
              - a project-map dict as produced by `aidev.repo_map.build_project_map`
                or `KnowledgeBase.build_project_map_full()`.

        max_neighbors:
            Soft cap on non-primary context files (dependencies, dependents,
            same-dir neighbors, configs).

        max_tests:
            Soft cap on test files.

        max_bytes_per_file:
            Maximum number of bytes to read for `content` per file. Files are
            decoded as UTF-8 with replacement and truncated if larger.

    Returns:
        A list of `ContextFile` instances, with the primary file first, followed
        by neighbors sorted by rough importance.
    """
    root = _project_root_from_kb_or_map(kb_or_map)
    rel = _normalize_rel(root, target_path)

    structure = _structure_from_kb_or_map(kb_or_map)
    kind = structure.get(rel, "other")
    language = _language_for_path(rel)

    # Choose summary + related-file implementation based on what we were given.
    if isinstance(kb_or_map, KnowledgeBase):
        kb = kb_or_map
        summary = _summary_from_kb(kb, rel)
        related = _get_related_from_kb(kb, rel)

        # Try to pull in tests/configs from contracts if possible.
        tests_from_contracts: List[str] = []
        configs_from_contracts: List[str] = []
        try:
            idx = kb.load_card_index()
            meta = idx.get(rel) or {}
            if isinstance(meta, dict):
                contracts = meta.get("contracts") or {}
                if isinstance(contracts, Mapping):
                    tns = contracts.get("test_neighbors") or []
                    if isinstance(tns, Sequence):
                        tests_from_contracts = [
                            t for t in tns if isinstance(t, str)
                        ]
                    cfg = contracts.get("config_contracts") or {}
                    if isinstance(cfg, Mapping):
                        cfg_paths = cfg.get("files") or []
                        if isinstance(cfg_paths, Sequence):
                            configs_from_contracts = [
                                c for c in cfg_paths if isinstance(c, str)
                            ]
        except Exception:
            pass

        # Merge tests from contracts into the related dict.
        tests_all = list(related.get("tests") or [])
        tests_all.extend(tests_from_contracts)
        related["tests"] = sorted(dict.fromkeys(tests_all))

        configs = sorted(dict.fromkeys(configs_from_contracts))
    else:
        project_map = kb_or_map
        summary = _summary_from_project_map(project_map, rel)
        # With only a project map, we cannot easily reconstruct graph edges,
        # so we fall back to same-dir neighbors only.
        related = {
            "same_dir": [],
            "dependencies": [],
            "dependents": [],
            "tests": [],
        }
        # same-dir heuristic via our synthesized structure
        same_dir: List[str] = []
        try:
            parent = Path(rel).parent
            for other in _structure_from_kb_or_map(project_map).keys():
                if other == rel:
                    continue
                if Path(other).parent == parent:
                    same_dir.append(other)
        except Exception:
            same_dir = []

        related["same_dir"] = sorted(dict.fromkeys(same_dir))
        configs = []  # project map does not encode config neighbors directly.

    # PRIMARY FILE
    primary_content, primary_size = _read_file_text(
        root, rel, max_bytes=max_bytes_per_file
    )
    primary = ContextFile(
        path=rel,
        role="primary",
        content=primary_content,
        summary=summary,
        language=language,
        kind=kind,
        size=primary_size,
        reason="File being edited.",
        weight=_role_weight("primary"),
    )

    # NEIGHBORS
    bundle: ContextBundle = [primary]
    seen: set[str] = {rel}

    def _add_paths(
        paths: Iterable[str],
        role: str,
        *,
        max_count: int,
        default_reason: str,
    ) -> None:
        nonlocal bundle, seen
        count = 0
        for p in paths:
            if count >= max_count:
                break
            if not isinstance(p, str):
                continue
            rp = _to_posix(p)
            if rp in seen:
                continue
            seen.add(rp)

            k = structure.get(rp, "other")
            lang = _language_for_path(rp)
            content, size = _read_file_text(
                root, rp, max_bytes=max_bytes_per_file
            )

            if isinstance(kb_or_map, KnowledgeBase):
                summ = _summary_from_kb(kb_or_map, rp)
            else:
                summ = _summary_from_project_map(kb_or_map, rp)

            cf = ContextFile(
                path=rp,
                role=role,
                content=content,
                summary=summ,
                language=lang,
                kind=k,
                size=size,
                reason=default_reason,
                weight=_role_weight(role),
            )
            bundle.append(cf)
            count += 1

    # Order: tests, dependencies, dependents, configs, same-dir neighbors.
    _add_paths(
        related.get("tests") or [],
        role="test",
        max_count=max_tests,
        default_reason="Likely tests for the primary file.",
    )
    remaining_neighbors = max_neighbors

    def _add_neighbors_group(
        key: str,
        role: str,
        reason: str,
    ) -> None:
        nonlocal remaining_neighbors
        if remaining_neighbors <= 0:
            return
        paths = related.get(key) or []
        if not paths:
            return
        before = len(bundle)
        _add_paths(
            paths,
            role=role,
            max_count=remaining_neighbors,
            default_reason=reason,
        )
        added = len(bundle) - before
        remaining_neighbors = max(0, remaining_neighbors - added)

    _add_neighbors_group(
        "dependencies",
        role="dependency",
        reason="File imported by the primary file.",
    )
    _add_neighbors_group(
        "dependents",
        role="dependent",
        reason="File that imports the primary file.",
    )

    if isinstance(kb_or_map, KnowledgeBase) and configs:
        _add_paths(
            configs,
            role="config",
            max_count=min(len(configs), max_neighbors),
            default_reason="Config file associated with the primary file.",
        )

    _add_neighbors_group(
        "same_dir",
        role="same_dir",
        reason="Neighbor in the same directory as the primary file.",
    )

    # Final ordering: prioritize files in the same logical "component family"
    # (same stem as the primary file) before falling back to the coarse role
    # weighting. This helps keep a React component, its CSS, and its tests
    # together as a tight unit.
    primary_only = bundle[0:1]
    rest = bundle[1:]

    primary_stem = Path(primary.path).stem

    def _family_boost(cf: ContextFile) -> float:
        return 15.0 if Path(cf.path).stem == primary_stem and cf.path != primary.path else 0.0

    rest.sort(
        key=lambda cf: (-(cf.weight + _family_boost(cf)), cf.path)
    )

    return primary_only + rest


# ---------------------------------------------------------------------------
# Public API — multi-path bundles
# ---------------------------------------------------------------------------


def build_context_bundle(
    target_path: str,
    kb_or_map: KbOrMap,
    *,
    max_neighbors: int = 16,
    max_tests: int = 8,
    max_bytes_per_file: int = 80_000,
) -> ContextBundle:
    """
    Alias for `build_context_for_edit`. Historically exported as
    `aidev.context.build_context_bundle`.
    """
    return build_context_for_edit(
        target_path,
        kb_or_map,
        max_neighbors=max_neighbors,
        max_tests=max_tests,
        max_bytes_per_file=max_bytes_per_file,
    )


def build_context_for_edit_from_project_map(
    target_path: str,
    project_map: Mapping[str, Any],
    *,
    max_neighbors: int = 16,
    max_tests: int = 8,
    max_bytes_per_file: int = 80_000,
) -> ContextBundle:
    """
    Convenience wrapper for callers that only have a project-map dict.
    """
    return build_context_for_edit(
        target_path,
        project_map,
        max_neighbors=max_neighbors,
        max_tests=max_tests,
        max_bytes_per_file=max_bytes_per_file,
    )


def build_context_bundle_for_paths(
    kb: KnowledgeBase | None,
    project_root: Path,
    paths: List[str],
    *,
    max_neighbors: int = 16,
    max_tests: int = 8,
    max_bytes_per_file: int = 80_000,
) -> Dict[str, ContextBundle]:
    """
    Build context bundles for multiple target paths in one call.

    This is the shared entrypoint for analyze and edit stages:

        - Analyze: use bundles to build high-signal previews / summaries.
        - Edit:    use bundles to derive `llm_payload_preview` and
                   `context_files` without re-implementing neighbor logic.

    Args:
        kb:
            Optional KnowledgeBase. When provided, we use its structure,
            summaries, and related-file graph. When None, we fall back to a
            minimal project-map-like dict using only `project_root`, which
            yields primary-file-only bundles (no graph neighbors).

        project_root:
            Filesystem project root. Used to normalize input paths and as the
            `root` value when we don't have a KnowledgeBase.

        paths:
            Collection of file paths to build bundles for. Paths may be abs or
            relative; they are normalized to project-relative POSIX.

        max_neighbors, max_tests, max_bytes_per_file:
            Passed through to `build_context_for_edit`. Tune these centrally
            to manage token footprint for GPT-5 / GPT-5-mini.

    Returns:
        A dict mapping normalized project-relative POSIX paths to their
        corresponding `ContextBundle` (primary + neighbors).
    """
    # Normalize and dedupe incoming paths early so we don't waste I/O or tokens.
    root = project_root.resolve()
    normalized: List[str] = []
    seen: set[str] = set()
    for p in paths:
        if not p:
            continue
        rel = _normalize_rel(root, p)
        if rel in seen:
            continue
        seen.add(rel)
        normalized.append(rel)

    # Choose the kb_or_map input for the underlying single-path builder.
    if kb is not None:
        kb_or_map: KbOrMap = kb
    else:
        # Minimal project-map variant: root only. This still lets callers
        # re-use the same read/summary pipeline, while keeping neighbor
        # discovery simple when we don't have a full KnowledgeBase yet.
        kb_or_map = {"root": str(root)}

    bundles: Dict[str, ContextBundle] = {}
    for rel in normalized:
        # `build_context_for_edit` is already careful about path normalization
        # and token footprint; we simply reuse it here.
        bundles[rel] = build_context_for_edit(
            rel,
            kb_or_map,
            max_neighbors=max_neighbors,
            max_tests=max_tests,
            max_bytes_per_file=max_bytes_per_file,
        )

    return bundles

def build_summary_context_files_for_path(
    *,
    kb: KnowledgeBase | None,
    project_root: Path,
    rel_path: str,
    max_neighbors: int = 8,
    max_tests: int = 4,
    max_bytes_per_file: int = 80_000,
) -> List[Dict[str, str]]:
    """
    Build a context_files list for a single primary file, suitable for
    summarize/analyze/edit payloads.

    Each entry is:
        { "path": "<repo-relative>", "snippet": "<short summary or content excerpt>" }
    """
    root = project_root.resolve()
    # Normalize to the same project-relative POSIX form used by the bundle builder
    rel_norm = _normalize_rel(root, str(rel_path))

    bundles = build_context_bundle_for_paths(
        kb=kb,
        project_root=root,
        paths=[rel_norm],
        max_neighbors=max_neighbors,
        max_tests=max_tests,
        max_bytes_per_file=max_bytes_per_file,
    )

    bundle = bundles.get(rel_norm, []) or []
    context_files = extract_neighbor_snippets(bundle, max_snippet_chars=800)
    return context_files

