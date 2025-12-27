# aidev/cleanup.py
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, List

def _cleanup_files(
    root: Path,
    *,
    includes: List[str] | None = None,
    excludes: List[str] | None = None,
    collapse_blank: bool = False,
    trim_trailing_ws: bool = True,
) -> int:
    """
    In-place cleanup. Returns #files changed.
    """
    includes = includes or ["**/*"]
    excludes = excludes or []
    changed = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if not any(fnmatch.fnmatch(rel, g) for g in includes):
            continue
        if any(fnmatch.fnmatch(rel, g) for g in excludes):
            continue
        try:
            old = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        new = old.replace("\r\n", "\n").replace("\r", "\n")
        if trim_trailing_ws:
            new = "\n".join(ln.rstrip() for ln in new.split("\n"))
        if collapse_blank:
            out: List[str] = []
            prev_blank = False
            for ln in new.split("\n"):
                blank = (ln.strip() == "")
                if blank and prev_blank:
                    continue
                out.append(ln)
                prev_blank = blank
            new = "\n".join(out)
        if new != old:
            p.write_text(new, encoding="utf-8")
            changed += 1
    return changed
