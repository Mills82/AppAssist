#!/usr/bin/env python3
"""End-to-end smoke runner for DevBotAPI.deep_research.

Runs deep research twice to validate:
- basic output invariants (ok/error, findings/evidence)
- cache miss then cache hit (or stable equivalent indicator)
- stable ordering of evidence IDs

Usage:
  python scripts/smoke_deep_research.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


QUERY = "How does intent routing work?"
DEPTH = "standard"


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _as_dict(obj: Any) -> Any:
    """Best-effort conversion of objects to dicts for introspection."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_as_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}

    # Common patterns: pydantic, dataclasses, custom .to_dict()
    for attr in ("model_dump", "dict", "to_dict", "as_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _as_dict(fn())
            except Exception:
                pass

    # Fallback to vars()
    try:
        return _as_dict(vars(obj))
    except Exception:
        return str(obj)


def _pick(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _find_cache_hit(d: Dict[str, Any]) -> Optional[bool]:
    """Search cache hit indicator in common places."""
    direct = _pick(d, ["cache_hit", "cached", "from_cache", "hit"], default=None)
    if isinstance(direct, bool):
        return direct

    cache_obj = _pick(d, ["cache", "cache_info", "caching", "meta", "metadata"], default=None)
    if isinstance(cache_obj, dict):
        v = _pick(cache_obj, ["cache_hit", "cached", "from_cache", "hit"], default=None)
        if isinstance(v, bool):
            return v

    return None


def _extract_evidence_items(raw: Any) -> List[Dict[str, Any]]:
    """Extract evidence items from either a list or a dict containing a list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        # common key variants
        ev = _pick(raw, ["evidence", "items", "refs", "references"], default=None)
        if isinstance(ev, list):
            return [x for x in ev if isinstance(x, dict)]
    return []


def _evidence_id(ev: Dict[str, Any]) -> str:
    v = _pick(ev, ["ev_id", "evidence_id", "id", "uid"], default=None)
    if v is None:
        return ""
    return str(v)


def _finding_evidence_refs(finding: Dict[str, Any]) -> List[str]:
    refs = _pick(
        finding,
        ["evidence_refs", "evidenceRefs", "evidence_ref_ids", "evidence_ids", "refs", "references"],
        default=None,
    )
    if isinstance(refs, list):
        return [str(x) for x in refs if x is not None and str(x).strip()]

    # Sometimes evidence items are embedded in the finding itself
    embedded = _pick(finding, ["evidence", "evidence_items"], default=None)
    ev_items = _extract_evidence_items(embedded)
    ids = [i for i in (_evidence_id(x) for x in ev_items) if i]
    if ids:
        return ids

    return []


def _gather_all_evidence(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a best-effort flat evidence list from top-level or nested fields."""
    top = _pick(result, ["evidence", "evidence_items"], default=None)
    top_items = _extract_evidence_items(top)
    if top_items:
        return top_items

    # Some APIs return evidence inside findings; collect and de-dup by id
    findings = _pick(result, ["findings", "results", "items"], default=[])
    all_items: List[Dict[str, Any]] = []
    if isinstance(findings, list):
        for f in findings:
            if not isinstance(f, dict):
                continue
            ev = _pick(f, ["evidence", "evidence_items"], default=None)
            all_items.extend(_extract_evidence_items(ev))

    # De-dup by id while preserving order
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for ev in all_items:
        eid = _evidence_id(ev)
        key = eid if eid else json.dumps(ev, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped


def _flatten_evidence_id_sequence(result: Dict[str, Any]) -> List[str]:
    """Stable sequence used to compare ordering across runs."""
    ev_items = _gather_all_evidence(result)
    ids = [i for i in (_evidence_id(ev) for ev in ev_items) if i]
    if ids:
        return ids

    # Fallback: flatten finding refs if no evidence items exist
    findings = _pick(result, ["findings", "results", "items"], default=[])
    seq: List[str] = []
    if isinstance(findings, list):
        for f in findings:
            if isinstance(f, dict):
                seq.extend(_finding_evidence_refs(f))
    return seq


def _normalize_result(raw_result: Any) -> Dict[str, Any]:
    result = _as_dict(raw_result)
    if not isinstance(result, dict):
        return {
            "_raw_type": str(type(raw_result)),
            "_raw": result,
            "ok": False,
            "error": "deep_research returned non-dict result; cannot validate",
        }

    ok = _pick(result, ["ok", "success", "is_ok"], default=None)
    if not isinstance(ok, bool):
        # Default ok to True if there's no error field and some content exists
        err = _pick(result, ["error", "err", "message"], default=None)
        ok = False if err else True

    normalized: Dict[str, Any] = {
        "repo_version": _pick(result, ["repo_version", "repoVersion", "version"], default=None),
        "cache_key": _pick(result, ["cache_key", "cacheKey", "key"], default=None),
        "cache_hit": _find_cache_hit(result),
        "ok": ok,
        "error": _pick(result, ["error", "err"], default=None),
        "findings": _pick(result, ["findings", "results", "items"], default=[]),
        "_result": result,
    }

    if not isinstance(normalized["findings"], list):
        normalized["findings"] = []

    normalized["evidence"] = _gather_all_evidence(result)

    return normalized


def _print_run(label: str, norm: Dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"repo_version: {norm.get('repo_version')}")
    print(f"cache_key: {norm.get('cache_key')}")
    print(f"cache_hit: {norm.get('cache_hit')}")
    print(f"ok: {norm.get('ok')}")
    print(f"error: {norm.get('error')}")

    evidence = norm.get("evidence")
    ev_count = len(evidence) if isinstance(evidence, list) else 0
    print(f"evidence_count: {ev_count}")

    findings = norm.get("findings")
    if not isinstance(findings, list):
        findings = []

    print("\nTop findings (up to 3):")
    for i, f in enumerate(findings[:3]):
        if not isinstance(f, dict):
            print(f"  {i+1}. <non-dict finding: {type(f)}>")
            continue
        title = _pick(f, ["title", "name", "summary"], default=None)
        confidence = _pick(f, ["confidence", "score", "probability"], default=None)
        if confidence is None:
            print(f"  {i+1}. {title}")
        else:
            print(f"  {i+1}. {title} (confidence={confidence})")

    print("\nEvidence items (first 3):")
    if isinstance(evidence, list):
        for i, ev in enumerate(evidence[:3]):
            if not isinstance(ev, dict):
                print(f"  {i+1}. <non-dict evidence: {type(ev)}>")
                continue
            ev_id = _evidence_id(ev)
            path = _pick(ev, ["path", "file", "filepath"], default=None)
            lines = _pick(ev, ["lines", "line_range", "range"], default=None)
            kind = _pick(ev, ["kind", "type"], default=None)
            print(f"  {i+1}. ev_id={ev_id} path={path} lines={lines} kind={kind}")


def _assert_invariants(norm1: Dict[str, Any], norm2: Dict[str, Any]) -> None:
    # Cache miss then hit
    ch1 = norm1.get("cache_hit")
    ch2 = norm2.get("cache_hit")
    if ch1 is None or ch2 is None:
        raise AssertionError(
            "Could not determine cache_hit indicator from result. "
            "Expected a boolean field like cache_hit/cached/from_cache at top-level or in meta/cache." 
            f"Got run1 cache_hit={ch1}, run2 cache_hit={ch2}."
        )
    if ch1 is not False:
        raise AssertionError(f"Expected first run cache_hit=False, got {ch1}")
    if ch2 is not True:
        raise AssertionError(f"Expected second run cache_hit=True, got {ch2}")

    # Stable evidence ordering
    seq1 = _flatten_evidence_id_sequence(norm1.get("_result", {}))
    seq2 = _flatten_evidence_id_sequence(norm2.get("_result", {}))
    if not seq1 or not seq2:
        raise AssertionError(
            "Could not extract evidence id sequence to compare ordering across runs. "
            "Expected evidence items with ev_id/id or findings with evidence_refs." 
            f"Got len(seq1)={len(seq1)} len(seq2)={len(seq2)}."
        )
    if seq1 != seq2:
        # show a small diff hint
        raise AssertionError(
            "Evidence ID ordering is not stable across runs (exact sequence mismatch).\n"
            f"run1(first 20)={seq1[:20]}\n"
            f"run2(first 20)={seq2[:20]}"
        )

    # Every finding has non-empty evidence refs
    findings = norm2.get("findings")
    if not isinstance(findings, list):
        findings = []
    failures: List[Tuple[int, str]] = []
    for idx, f in enumerate(findings):
        if not isinstance(f, dict):
            failures.append((idx, "finding is not a dict"))
            continue
        refs = _finding_evidence_refs(f)
        if not refs:
            title = _pick(f, ["title", "name", "summary"], default="<no title>")
            failures.append((idx, f"empty evidence refs for finding: {title}"))

    if failures:
        msg = "\n".join([f"  - finding[{i}]: {reason}" for i, reason in failures])
        raise AssertionError(
            "One or more findings missing non-empty evidence_refs (or equivalent).\n" + msg
        )


def _import_devbot_api():
    # Prefer aidev.assistant_api; fall back to assistant_api
    try:
        from aidev.assistant_api import DevBotAPI  # type: ignore

        return DevBotAPI
    except Exception as e1:
        try:
            from assistant_api import DevBotAPI  # type: ignore

            _eprint(
                "[info] Imported DevBotAPI from assistant_api (fallback); "
                "consider using aidev.assistant_api if available."
            )
            return DevBotAPI
        except Exception as e2:
            raise ImportError(
                "Failed to import DevBotAPI from aidev.assistant_api and assistant_api. "
                f"aidev.assistant_api error: {e1}; assistant_api error: {e2}"
            )


def run_deep_research(api: Any) -> Dict[str, Any]:
    raw = api.deep_research(query=QUERY, depth=DEPTH)
    return _normalize_result(raw)


def main() -> int:
    try:
        DevBotAPI = _import_devbot_api()
    except Exception as e:
        _eprint(str(e))
        return 2

    try:
        api = DevBotAPI()  # do not assume constructor signature
    except Exception as e:
        _eprint(f"Failed to instantiate DevBotAPI() with no args: {e}")
        return 2

    try:
        norm1 = run_deep_research(api)
        _print_run("Run 1", norm1)

        time.sleep(0.5)

        norm2 = run_deep_research(api)
        _print_run("Run 2", norm2)

        _assert_invariants(norm1, norm2)
        print("\nSmoke deep_research: PASS")
        return 0
    except AssertionError as e:
        _eprint("\nSmoke deep_research: FAIL")
        _eprint(str(e))
        return 1
    except Exception as e:
        _eprint("\nSmoke deep_research: ERROR")
        _eprint(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
