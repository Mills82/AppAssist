# aidev/llm_io.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

@dataclass
class CodeEdit:
    path: str          # relative path (under project root)
    content: str       # full file content to write
    rec_id: str = ""   # optional logical record id for grouping/summarizing

def fetch_code_jsonl(p: Path | str) -> Iterator[CodeEdit]:
    """
    Read a JSONL file containing objects like:
      {"path":"src/app.py","content":"...","rec_id":"card-42"}
    and yield CodeEdit entries. Lines that fail to parse are ignored.
    """
    path = Path(p)
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield CodeEdit(
                    path=obj.get("path", ""),
                    content=obj.get("content", ""),
                    rec_id=obj.get("rec_id", "") or "",
                )
            except Exception:
                continue
