#!/usr/bin/env python3
# audit_files.py
# Usage examples:
#   python audit_files.py --root C:\Users\mattm\Repos\Metaverse
#   python audit_files.py --root C:\Users\mattm\Repos\Metaverse --format csv
#   python audit_files.py --root C:\Users\mattm\Repos\Metaverse --preview-if-bytes-lt 2000
#   python audit_files.py --root . --only missing
#   python audit_files.py --root . --sort-by chars --desc

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

def read_text_safe(path: Path, max_bytes: Optional[int] = None) -> Tuple[str, bool]:
    """Read text with utf-8 then latin-1 fallback; optionally limit bytes."""
    try:
        data = path.read_bytes() if max_bytes is None else path.read_bytes()[:max_bytes]
    except Exception:
        return ("", False)
    try:
        return (data.decode("utf-8", errors="replace"), True)
    except Exception:
        try:
            return (data.decode("latin-1", errors="replace"), True)
        except Exception:
            return ("", False)

def count_chars_lines(path: Path, max_bytes_for_charcount: Optional[int]) -> Tuple[int, int, bool]:
    """
    Return (char_count, line_count, ok).
    If file is larger than max_bytes_for_charcount (and it's set), skip and return ok=False.
    """
    try:
        if (max_bytes_for_charcount is not None) and path.stat().st_size > max_bytes_for_charcount:
            return (0, 0, False)
        text, ok = read_text_safe(path, None)
        if not ok:
            return (0, 0, False)
        char_count = len(text)
        if not text:
            line_count = 0
        else:
            line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
        return (char_count, line_count, True)
    except Exception:
        return (0, 0, False)

def format_row_text(r: dict) -> str:
    exists = "✅" if r["exists"] else "❌"
    size = r["bytes"]
    chars = r["chars"] if r["chars_ok"] else "-"
    lines = r["lines"] if r["chars_ok"] else "-"
    typ = r.get("type", "")
    return f"{exists}  {r['path']}\n    type={typ:12} bytes={size:<8} chars={chars:<8} lines={lines:<8} mtime={r['mtime']}\n"

def main():
    ap = argparse.ArgumentParser(description="Audit files listed in project_structure.json")
    ap.add_argument("--root", default=".", help="Project root directory")
    ap.add_argument("--manifest", default="project_structure.json",
                    help="Path to project_structure.json (relative to --root or absolute)")
    ap.add_argument("--format", choices=["text", "csv", "json"], default="text", help="Output format")
    ap.add_argument("--only", choices=["all", "missing", "empty"], default="all",
                    help="Filter to only missing or empty (0-byte) files")
    ap.add_argument("--sort-by", choices=["path", "bytes", "chars", "lines"], default="path", help="Sort output")
    ap.add_argument("--desc", action="store_true", help="Sort descending")
    ap.add_argument("--max-bytes-for-charcount", type=int, default=2_000_000,
                    help="Skip char/line counting for files larger than this. Set to -1 to always count.")
    ap.add_argument("--preview-if-bytes-lt", dest="preview_if_bytes_lt", type=int, default=0,
                    help="If >0, print file contents for files smaller than this many bytes")
    ap.add_argument("--type", choices=["source", "config", "docs", "build-config"],
                    help="Filter by file type from manifest (optional)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read manifest JSON: {e}", file=sys.stderr)
        sys.exit(1)

    max_bytes_for_charcount = None if args.max_bytes_for_charcount == -1 else args.max_bytes_for_charcount

    results = []
    for rel_path, ftype in manifest.items():
        p = (root / rel_path).resolve()
        exists = p.exists() and p.is_file()
        size = p.stat().st_size if exists else 0
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if exists else "-"
        chars = lines = 0
        chars_ok = False
        if exists:
            chars, lines, chars_ok = count_chars_lines(p, max_bytes_for_charcount)

        rec = {
            "path": rel_path,
            "abspath": str(p) if exists else "",
            "exists": exists,
            "bytes": size,
            "chars": chars,
            "lines": lines,
            "chars_ok": chars_ok,
            "mtime": mtime,
            "type": ftype
        }
        results.append(rec)

    # Filtering
    if args.type:
        results = [r for r in results if r.get("type") == args.type]
    if args.only == "missing":
        results = [r for r in results if not r["exists"]]
    elif args.only == "empty":
        results = [r for r in results if r["exists"] and r["bytes"] == 0]

    # Sorting
    key_map = {
        "path": lambda r: r["path"].lower(),
        "bytes": lambda r: r["bytes"],
        "chars": lambda r: r["chars"],
        "lines": lambda r: r["lines"],
    }
    results.sort(key=key_map[args.sort_by], reverse=args.desc)

    # Output
    if args.format == "json":
        print(json.dumps(results, indent=2))
    elif args.format == "csv":
        print("path,type,exists,bytes,chars,lines,mtime")
        for r in results:
            print(f"{r['path']},{r.get('type','')},{int(r['exists'])},{r['bytes']},"
                  f"{r['chars'] if r['chars_ok'] else ''},{r['lines'] if r['chars_ok'] else ''},{r['mtime']}")
    else:
        # text
        missing = sum(1 for r in results if not r["exists"])
        empty = sum(1 for r in results if r["exists"] and r["bytes"] == 0)
        total = len(results)
        print(f"Scanned {total} files (missing={missing}, empty-bytes={empty})\n")
        for r in results:
            print(format_row_text(r))
            if args.preview_if_bytes_lt and r["exists"] and r["bytes"] <= args.preview_if_bytes_lt:
                text, ok = read_text_safe(Path(root / r["path"]), None)
                if ok:
                    print("----- FILE PREVIEW BEGIN -----")
                    preview = text if len(text) <= 4000 else text[:4000] + "\n...[truncated]..."
                    print(preview)
                    print("----- FILE PREVIEW END -----\n")

if __name__ == "__main__":
    main()
