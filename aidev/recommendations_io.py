# aidev/recommendations_io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

RECS_PATH = Path(".aidev/recommendations.json")


def _normalize_action(action: Any) -> Dict[str, Any]:
    """Normalize a single action into a canonical edit dict.

    We accept a variety of legacy shapes and return a dict with at least
    a 'path' key where possible, while preserving FileEdit-style fields:

      - path
      - patch_unified / patch / diff
      - content
      - rec_id, summary, etc.
    """
    if isinstance(action, dict):
        # If it already looks like a modern FileEdit (has path plus either
        # content/patch_unified/patch/diff), just preserve it with minimal
        # cleanup instead of collapsing fields.
        if ("path" in action or "file" in action) and any(
            k in action for k in ("patch_unified", "patch", "diff", "content")
        ):
            path = action.get("path") or action.get("file")
            out: Dict[str, Any] = {"path": path}

            # Preserve all relevant edit fields distinctly
            for key in ("patch_unified", "patch", "diff", "content"):
                if key in action:
                    out[key] = action[key]

            # Preserve any extra metadata (rec_id, summary, why, etc.)
            for k, v in action.items():
                if k in ("path", "file"):
                    continue
                if k not in out:
                    out[k] = v
            return out

        # Legacy shapes that might not follow FileEdit yet: fall back to the older behavior,
        # but DO NOT treat content as a diff.
        if "path" in action or "file" in action:
            path = action.get("path") or action.get("file")
            # Only unify true diff-style fields; keep content separate.
            patch = action.get("patch") or action.get("diff") or action.get("patch_unified")
            out: Dict[str, Any] = {"path": path, "patch": patch}

            # Preserve content (if present) and any extra metadata.
            if "content" in action:
                out["content"] = action["content"]

            for k, v in action.items():
                if k in ("path", "file", "patch", "diff", "patch_unified", "content"):
                    continue
                if k not in out:
                    out[k] = v
            return out

        # Plain dict that doesn't look like an edit; just copy it.
        return dict(action)

    if isinstance(action, (list, tuple)):
        # common legacy shape: (path, patch)
        try:
            if len(action) == 2:
                return {"path": action[0], "patch": action[1]}
        except Exception:
            pass
        return {"value": list(action)}

    if isinstance(action, str):
        return {"path": action, "patch": None}

    return {"value": action}


def _normalize_rec(rec: Any, default_index: int = 1) -> Dict[str, Any]:
    """Normalize a single recommendation-like object into a canonical shape.

    Guarantees at minimum these top-level keys: id, title, reason, summary,
    edits (list). This helps downstream code rely on a single shape and avoids
    unpacking errors from unexpected legacy formats.
    """
    out: Dict[str, Any] = {}
    if not isinstance(rec, dict):
        # Minimal coercion for simple types
        out["id"] = f"rec-{default_index}"
        out["title"] = str(rec)
        out["reason"] = ""
        out["summary"] = str(rec)[:300]
        out["edits"] = []
        return out

    out["id"] = rec.get("id") or rec.get("recommendation_id") or f"rec-{default_index}"
    out["title"] = rec.get("title") or rec.get("rationale") or rec.get("name") or "Untitled"
    out["reason"] = rec.get("reason") or rec.get("why") or rec.get("explanation") or ""
    out["summary"] = (rec.get("summary") or out["reason"] or out["title"])[:300]
    out["risk"] = rec.get("risk", "medium")
    out["schema_version"] = rec.get("schema_version", 2)

    # Normalize actions -> edits
    raw_actions = []
    if "edits" in rec and isinstance(rec.get("edits"), (list, tuple)):
        raw_actions = list(rec.get("edits"))
    elif "diffs" in rec and isinstance(rec.get("diffs"), (list, tuple)):
        raw_actions = list(rec.get("diffs"))
    elif "actions" in rec and isinstance(rec.get("actions"), (list, tuple)):
        raw_actions = list(rec.get("actions"))
    else:
        # Some legacy blobs may put file changes under 'files' or 'files_modified'
        if isinstance(rec.get("files"), list):
            raw_actions = rec.get("files")
        elif isinstance(rec.get("files_modified"), list):
            raw_actions = rec.get("files_modified")

    edits: List[Dict[str, Any]] = []
    for a in raw_actions:
        normalized = _normalize_action(a)
        # ensure minimal structure
        if "path" not in normalized and "value" not in normalized:
            # if it looks like a recommendation (nested), try to extract edits field
            if isinstance(a, dict) and "edits" in a and isinstance(a["edits"], list):
                for sub in a["edits"]:
                    edits.append(_normalize_action(sub))
                continue
        edits.append(normalized)

    out["edits"] = edits
    # keep actions for backwards compatibility, but normalized
    out["actions"] = edits
    # mirror into 'diffs' as alias (some code expects diffs)
    out["diffs"] = edits

    # preserve other metadata conservatively
    for k, v in rec.items():
        if k in ("id", "title", "reason", "summary", "risk", "schema_version", "edits", "actions", "diffs", "files", "files_modified"):
            continue
        out.setdefault(k, v)

    return out


def _normalize(rec_blob: Any) -> List[Dict[str, Any]]:
    """
    Accept either:
      - v2 array (canonical)
      - legacy object with {"rationales":[{...,"actions":[...]}, ...]}
      - single recommendation dict
    Return a v2-style list[Recommendation] with normalized 'edits' arrays.
    """
    if isinstance(rec_blob, list):
        out: List[Dict[str, Any]] = []
        for i, r in enumerate(rec_blob, start=1):
            out.append(_normalize_rec(r, default_index=i))
        return out

    if isinstance(rec_blob, dict) and "rationales" in rec_blob:
        out: List[Dict[str, Any]] = []
        for i, r in enumerate(rec_blob.get("rationales", []), start=1):
            # support legacy keys mapping
            mapped = {
                "schema_version": 2,
                "id": r.get("id", f"rec-{i}"),
                "rationale": r.get("title", "General"),
                "title": r.get("title", "Untitled"),
                "reason": r.get("why", ""),
                "summary": r.get("summary", r.get("why", ""))[:300],
                "risk": r.get("risk", "medium"),
                "budget_estimate": r.get("budget_estimate"),
                "files": r.get("files", []),
                "actions": r.get("actions", []),
                "acceptance_criteria": r.get("acceptance_criteria", []),
            }
            out.append(_normalize_rec(mapped, default_index=i))
        return out

    # If it's a single recommendation dict-like object, normalize it into a list
    if isinstance(rec_blob, dict):
        return [_normalize_rec(rec_blob, default_index=1)]

    # Unknown shape -> empty (fail-soft)
    return []


def save_recommendations(rec_blob: Any) -> List[Dict[str, Any]]:
    recs = _normalize(rec_blob)
    RECS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECS_PATH.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    return recs


def load_recommendations() -> List[Dict[str, Any]]:
    if not RECS_PATH.exists():
        return []
    try:
        data = json.loads(RECS_PATH.read_text(encoding="utf-8"))
        return _normalize(data)
    except Exception:
        # Fail-soft: return empty list on any error to avoid crashes in pipelines
        return []
