"""Release the rendered report through a local git repository.

After the report stage writes ``<report-dir>``, :func:`run_publish`:

1. initializes a git repository there when none exists (branch ``main``,
   ``core.autocrlf=false`` so commits hold the rendered bytes);
2. writes the GitHub Pages workflow (:mod:`bench.publish.workflow`);
3. summarizes the working-tree changes into a generated commit message;
4. asks ``confirm`` for approval, and only on approval stages everything,
   commits, and pushes to the upstream, ``origin``, or the only remote. A
   clean tree with unpushed commits offers to push them instead.

A declined prompt stages nothing and leaves every file in place. The repository
never needs a remote: without one, the commit stays local with a note. The
report runner keeps ``.git`` and ``.github`` across re-renders.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

from ..config import RunConfig
from .errors import PublishError
from .workflow import PAGES_WORKFLOW, WORKFLOW_PATH

# GitHub rejects pushes that carry a file above 100 MB.
MAX_FILE_BYTES = 100 * 1024 * 1024
# Changed-area lines listed in the commit body before the rest is summarized.
MAX_AREA_LINES = 20
BRANCH = "main"

Status = Literal["nothing_to_publish", "declined", "published", "pushed"]


@dataclass
class PublishResult:
    status: Status
    repo_dir: str
    created_repo: bool = False
    message: str = ""
    commit: str = ""
    push_target: str = ""
    notes: list[str] = field(default_factory=list)


def publish_enabled(cfg: RunConfig) -> bool:
    return bool((cfg.report.get("publish") or {}).get("enabled"))


def _git(repo: Path, *args: str, stdin: str | None = None, check: bool = True):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise PublishError(f"`git {' '.join(args)}` failed in '{repo}': {detail}")
    return proc


def _ensure_repo(repo: Path) -> bool:
    """Initialize ``repo`` when it has no ``.git``; return whether one was created."""
    created = not (repo / ".git").exists()
    if created:
        _git(repo, "init", "-b", BRANCH)
    # Guard against committing to an enclosing repository by mistake.
    top = _git(repo, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(top).resolve() != repo.resolve():
        raise PublishError(
            f"'{repo}' resolves to the git repository at '{top}', not its own; "
            "refusing to publish."
        )
    _git(repo, "config", "core.autocrlf", "false")
    return created


def _write_workflow(repo: Path) -> None:
    path = repo / WORKFLOW_PATH
    expected = PAGES_WORKFLOW.encode("utf-8")
    try:
        if path.is_file() and path.read_bytes() == expected:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    except OSError as exc:
        raise PublishError(f"could not write the Pages workflow '{path}': {exc}") from exc


def _changes(repo: Path) -> list[tuple[str, str]]:
    """Return ``(kind, path)`` pairs, ``kind`` in added/modified/deleted/ignored."""
    out = _git(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored"
    ).stdout
    fields = out.split("\0")
    changes: list[tuple[str, str]] = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            i += 1  # the rename or copy source follows as its own field
        if code == "!!":
            kind = "ignored"
        elif code == "??" or "A" in code:
            kind = "added"
        elif "D" in code:
            kind = "deleted"
        else:
            kind = "modified"
        changes.append((kind, path))
    return changes


def _area(path: str) -> str:
    parts = PurePosixPath(path).parts
    if len(parts) == 1:
        return path
    depth = 2 if len(parts) > 2 else 1
    return "/".join(parts[:depth]) + "/"


def _counts_text(counts: Counter) -> str:
    return ", ".join(f"{counts[k]} {k}" for k in ("added", "modified", "deleted") if counts[k])


def _title(cfg: RunConfig) -> str:
    return cfg.report.get("title") or cfg.friendly_name or cfg.name


def commit_message(cfg: RunConfig, changes: list[tuple[str, str]]) -> str:
    """The generated commit message: subject, totals, and changed areas."""
    areas: dict[str, Counter] = {}
    for kind, path in changes:
        areas.setdefault(_area(path), Counter())[kind] += 1
    lines = [f"Update {_title(cfg)}", "", _counts_text(Counter(k for k, _ in changes)) + ".", ""]
    names = sorted(areas)
    lines += [f"- {name} ({_counts_text(areas[name])})" for name in names[:MAX_AREA_LINES]]
    if len(names) > MAX_AREA_LINES:
        lines.append(f"- and {len(names) - MAX_AREA_LINES} more areas")
    return "\n".join(lines) + "\n"


def _oversized(repo: Path, changes: list[tuple[str, str]]) -> list[str]:
    big = []
    for kind, path in changes:
        target = repo / path
        if kind != "deleted" and target.is_file() and target.stat().st_size > MAX_FILE_BYTES:
            big.append(f"{path} ({target.stat().st_size / 1024 / 1024:.0f} MB)")
    return big


def _push_plan(repo: Path) -> tuple[list[str], str, int]:
    """Return the push arguments, their target, and the unpushed commit count.

    Pushes to the branch's upstream, naming its remote and destination ref so
    ``push.default`` and branch-name differences cannot redirect the push. Else
    sets an upstream on ``origin`` or on the only remote. Empty arguments mean
    there is no commit or no remote to push to.
    """
    if _git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False).returncode != 0:
        return [], "", 0
    branch = _git(repo, "symbolic-ref", "-q", "--short", "HEAD", check=False).stdout.strip()
    if branch:
        up_remote = _git(repo, "config", f"branch.{branch}.remote", check=False).stdout.strip()
        up_merge = _git(repo, "config", f"branch.{branch}.merge", check=False).stdout.strip()
        if up_remote and up_merge:
            ahead = _git(repo, "rev-list", "--count", "@{u}..HEAD", check=False).stdout.strip()
            short = up_merge.removeprefix("refs/heads/")
            return (
                ["push", up_remote, f"HEAD:{up_merge}"],
                f"{up_remote}/{short}",
                int(ahead or 0),
            )
    remotes = _git(repo, "remote").stdout.split()
    remote = "origin" if "origin" in remotes else (remotes[0] if len(remotes) == 1 else "")
    if not remote:
        return [], "", 0
    ahead = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    return ["push", "-u", remote, "HEAD"], remote, int(ahead or 0)


def _push(repo: Path, notes: list[str]) -> str:
    """Push the local commits; return the target, or an empty string when skipped."""
    args, target, _ = _push_plan(repo)
    if not args:
        remotes = _git(repo, "remote").stdout.split()
        if remotes:
            notes.append(
                f"no upstream and no `origin` among remotes {remotes}; the commit is local. "
                f"Run `git -C \"{repo}\" push -u <remote> HEAD` once."
            )
        else:
            notes.append(
                f"no remote; the commit is local. Run `git -C \"{repo}\" remote add origin "
                "<url>` to push next time."
            )
        return ""
    proc = _git(repo, *args, check=False)
    if proc.returncode != 0:
        raise PublishError(
            f"push to {target} failed; the commit is kept locally in '{repo}': "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return target


def _check_identity(repo: Path) -> None:
    for var in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        if _git(repo, "var", var, check=False).returncode != 0:
            raise PublishError(
                "git has no commit identity; nothing was staged. Run "
                "`git config --global user.name \"Your Name\"` and "
                "`git config --global user.email you@example.com`."
            )


def run_publish(
    cfg: RunConfig, report_dir: str | Path, confirm: Callable[[str, str], bool]
) -> PublishResult:
    """Commit and push the report at ``report_dir`` once ``confirm`` approves.

    ``confirm(summary, question)`` shows the summary and returns the answer.
    """
    if shutil.which("git") is None:
        raise PublishError(
            "report.publish needs git on PATH. Install it from https://git-scm.com/downloads."
        )
    repo = Path(report_dir)
    if not repo.is_dir():
        raise PublishError(f"report directory '{repo}' does not exist; render the report first.")

    created = _ensure_repo(repo)
    _write_workflow(repo)
    result = PublishResult(status="nothing_to_publish", repo_dir=str(repo), created_repo=created)
    header = f"Report repository: {repo}" + (" (new)" if created else "")

    changes = _changes(repo)
    ignored = [path for kind, path in changes if kind == "ignored"]
    if ignored:
        more = f", and {len(ignored) - 5} more" if len(ignored) > 5 else ""
        raise PublishError(
            f"{len(ignored)} report file(s) match a gitignore rule and would be left out "
            "of the site (check `git config core.excludesFile`); nothing was staged: "
            + ", ".join(ignored[:5]) + more
        )
    if not changes:
        # A clean tree can still hold commits whose push failed on an earlier run.
        _, target, ahead = _push_plan(repo)
        if ahead and confirm(
            f"{header}\n\nNo new changes; {ahead} local commit(s) not yet pushed to {target}.",
            "Push? [y/N] ",
        ):
            result.push_target = _push(repo, result.notes)
            result.status = "pushed"
        elif ahead:
            result.status = "declined"
        return result
    _check_identity(repo)
    big = _oversized(repo, changes)
    if big:
        raise PublishError(
            "GitHub rejects files over 100 MB; nothing was staged. Too large: " + ", ".join(big)
        )

    result.message = commit_message(cfg, changes)
    if not confirm(f"{header}\n\n{result.message}", "Stage, commit, and push? [y/N] "):
        result.status = "declined"
        return result

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-F", "-", stdin=result.message)
    result.commit = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    result.status = "published"
    result.push_target = _push(repo, result.notes)
    return result
