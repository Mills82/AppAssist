# aidev/vcs/commit.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional


def _run_git(args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


def is_git_repo(root: Path) -> bool:
    try:
        res = _run_git(["rev-parse", "--is-inside-work-tree"], root)
        return res.returncode == 0 and res.stdout.strip() == "true"
    except Exception:
        return False


def commit_all(
    root: Path,
    *,
    message: str,
    signoff: bool = False,
    allow_empty: bool = False,
) -> Dict[str, str]:
    add = _run_git(["add", "-A"], root)
    if add.returncode != 0:
        return {
            "stage": "add",
            "stderr": add.stderr.strip(),
            "stdout": add.stdout.strip(),
            "ok": "false",
        }

    args = ["commit", "-m", message]
    if allow_empty:
        args.insert(1, "--allow-empty")
    if signoff:
        args.insert(1, "--signoff")

    com = _run_git(args, root)
    return {
        "stage": "commit",
        "returncode": str(com.returncode),
        "stdout": com.stdout.strip(),
        "stderr": com.stderr.strip(),
        "ok": "true" if com.returncode == 0 else "false",
    }


def push(
    root: Path,
    *,
    remote: str = "origin",
    branch: Optional[str] = None,
) -> Dict[str, str]:
    # Resolve current branch if not supplied
    if branch is None:
        br = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
        branch = br.stdout.strip() if br.returncode == 0 else None

    if not branch or branch == "HEAD":
        # Detached HEAD or unknown; try simple push
        args = ["push", remote]
    else:
        args = ["push", remote, branch]

    res = _run_git(args, root)
    return {
        "stage": "push",
        "returncode": str(res.returncode),
        "stdout": res.stdout.strip(),
        "stderr": res.stderr.strip(),
        "ok": "true" if res.returncode == 0 else "false",
    }


def git_snapshot(
    root: Path | str,
    *,
    message: Optional[str] = None,
    commit_message: Optional[str] = None,
    push: bool = False,
    remote: str = "origin",
    branch: Optional[str] = None,
    signoff: bool = False,
    allow_empty: bool = False,
) -> Dict[str, str]:
    """
    - no-op if not a git repo
    - git add -A && git commit -m "<message>" [--allow-empty] [--signoff]
    - optional push

    You can provide either:
      - commit_message: preferred, or
      - message: kept for backward compatibility.

    If neither is provided, a generic "aidev: snapshot" message is used.

    Returns a dict describing the last step executed.
    """
    rootp = Path(root).resolve()
    if not is_git_repo(rootp):
        return {"stage": "noop", "ok": "false", "stderr": "not a git repo"}

    # Prefer commit_message if present, then message, then a default.
    msg = commit_message or message or "aidev: snapshot"

    result = commit_all(rootp, message=msg, signoff=signoff, allow_empty=allow_empty)
    if result.get("ok") != "true":
        return result

    if push:
        push_changes = push_to_remote(rootp, remote=remote, branch=branch)
        return push_changes
    return result


# small alias (helps keep public API stable if we add extra logic later)
def push_to_remote(root: Path, *, remote: str = "origin", branch: Optional[str] = None) -> Dict[str, str]:
    return push(root, remote=remote, branch=branch)
