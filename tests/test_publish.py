"""Report publishing: ``bench.publish`` plus the CLI publish hooks.

``run_publish`` turns the rendered report directory into its own git
repository, writes the GitHub Pages workflow, generates a commit message from
the working-tree changes, and commits (and pushes) only after ``confirm``
approves. These tests drive real ``git`` on tiny synthetic report trees under
``tmp_path``; every git invocation runs with an isolated environment so the
developer's global or system git config cannot leak in. The whole module skips
when ``git`` is not on PATH.

The CLI helpers are exercised directly: ``_confirm_publish`` must decline on a
non-TTY stdin without touching ``input``, and ``_publish_after_report`` must be
a no-op (exit 0, no repository) for ``--no-publish`` and for a disabled
``report.publish`` block.
"""

from __future__ import annotations

import shutil
import subprocess
import types
from pathlib import Path

import pytest

import bench.publish.runner as publish_runner
from bench.publish import PublishError, commit_message, publish_enabled, run_publish
from bench.publish.workflow import PAGES_WORKFLOW

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


# ── fixtures / helpers ───────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def isolated_git_env(tmp_path, monkeypatch):
    """Pin identity and config so host git settings cannot affect the tests."""
    empty_global_config = tmp_path / "empty-gitconfig"
    empty_global_config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test Committer")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "committer@example.com")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_global_config))


def _cfg(friendly_name: str = "My Run") -> types.SimpleNamespace:
    """Stand-in config: run_publish only reads report, friendly_name and name.

    ``report`` carries no ``title``, so commit subjects fall back to
    ``friendly_name`` (see bench.publish.runner._title).
    """
    return types.SimpleNamespace(
        report={"publish": {"enabled": True}},
        friendly_name=friendly_name,
        name="civbench-run",
    )


def _fake_report(report_dir: Path) -> Path:
    """A minimal rendered report tree: index.html plus one analysis asset."""
    (report_dir / "assets" / "bt_main").mkdir(parents=True)
    (report_dir / "index.html").write_text(
        "<!doctype html><title>report</title>\n", encoding="utf-8"
    )
    (report_dir / "assets" / "bt_main" / "ratings.csv").write_text(
        "player,rating\nVanilla,1500\n", encoding="utf-8"
    )
    return report_dir


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run ``git -C repo ...`` and return stdout; raise on a failed call."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8",
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"`git {' '.join(args)}` failed: {proc.stderr.strip()}")
    return proc.stdout


def _bare_repo(path: Path) -> Path:
    subprocess.run(
        ["git", "init", "--bare", str(path)], check=True, capture_output=True, text=True
    )
    return path


# ── run_publish: pure helpers ────────────────────────────────────────────────
def test_publish_enabled_reads_the_report_block():
    assert publish_enabled(_cfg()) is True
    assert publish_enabled(types.SimpleNamespace(report={})) is False
    assert publish_enabled(types.SimpleNamespace(report={"publish": None})) is False
    assert publish_enabled(
        types.SimpleNamespace(report={"publish": {"enabled": False}})
    ) is False


def test_commit_message_subject_and_totals_format():
    # The exact body line _counts_text produces: "added, modified, deleted"
    # order, only non-zero kinds, period appended by commit_message.
    changes = [
        ("added", "notes.txt"),
        ("modified", "index.html"),
        ("deleted", "assets/bt_main/ratings.csv"),
    ]
    message = commit_message(_cfg("My Run"), changes)
    assert message.splitlines()[0] == "Update My Run"
    assert "1 added, 1 modified, 1 deleted." in message


# ── run_publish: repository lifecycle ────────────────────────────────────────
def test_first_publish_declined_initializes_repo_and_stages_nothing(tmp_path):
    report_dir = _fake_report(tmp_path / "report")
    seen: dict[str, str] = {}

    def confirm(summary: str, question: str) -> bool:
        seen["summary"] = summary
        return False

    result = run_publish(_cfg("My Run"), report_dir, confirm)

    assert result.status == "declined"
    assert result.created_repo is True
    assert (report_dir / ".git").is_dir()
    workflow = report_dir / ".github" / "workflows" / "pages.yml"
    assert workflow.is_file()
    assert workflow.read_text(encoding="utf-8") == PAGES_WORKFLOW
    # A declined prompt commits nothing and touches nothing.
    assert _git(report_dir, "diff", "--cached", "--name-only").strip() == ""
    assert (report_dir / "index.html").is_file()
    assert (report_dir / "assets" / "bt_main" / "ratings.csv").is_file()
    assert "Update My Run" in seen["summary"]


def test_approved_publish_pushes_to_origin_bare(tmp_path):
    bare = _bare_repo(tmp_path / "origin.git")
    report_dir = _fake_report(tmp_path / "report")
    # Pre-create the repo so origin exists before run_publish's push step.
    subprocess.run(
        ["git", "init", "-b", "main", str(report_dir)],
        check=True, capture_output=True, text=True,
    )
    _git(report_dir, "remote", "add", "origin", str(bare))

    result = run_publish(_cfg("My Run"), report_dir, lambda summary, question: True)

    assert result.status == "published"
    assert result.push_target == "origin"
    assert result.commit
    subjects = _git(bare, "log", "--format=%s", "main").splitlines()
    assert subjects[0] == "Update My Run"


def test_approved_publish_without_remote_keeps_commit_local(tmp_path):
    report_dir = _fake_report(tmp_path / "report")

    result = run_publish(_cfg("My Run"), report_dir, lambda summary, question: True)

    assert result.status == "published"
    assert result.push_target == ""
    assert len(result.notes) == 1
    assert "remote add origin" in result.notes[0]
    assert _git(report_dir, "rev-parse", "HEAD").strip() == _git(
        report_dir, "rev-parse", result.commit
    ).strip()


def test_second_publish_without_changes_never_asks(tmp_path):
    report_dir = _fake_report(tmp_path / "report")
    first = run_publish(_cfg("My Run"), report_dir, lambda summary, question: True)
    assert first.status == "published"

    calls: list[str] = []
    second = run_publish(_cfg("My Run"), report_dir, lambda s, q: calls.append(s) or True)

    assert second.status == "nothing_to_publish"
    assert second.created_repo is False
    assert calls == []


def test_mixed_changes_message_via_declined_run(tmp_path):
    report_dir = _fake_report(tmp_path / "report")
    # Commit a baseline that already carries the Pages workflow, so the next
    # _write_workflow is a no-op and only our three edits show up as changes.
    workflow = report_dir / ".github" / "workflows" / "pages.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_bytes(PAGES_WORKFLOW.encode("utf-8"))
    subprocess.run(
        ["git", "init", "-b", "main", str(report_dir)],
        check=True, capture_output=True, text=True,
    )
    _git(report_dir, "add", "-A")
    _git(report_dir, "commit", "-q", "-m", "baseline")

    (report_dir / "index.html").write_text("<html>changed</html>\n", encoding="utf-8")
    (report_dir / "assets" / "bt_main" / "ratings.csv").unlink()
    (report_dir / "notes.txt").write_text("new file\n", encoding="utf-8")

    result = run_publish(_cfg("My Run"), report_dir, lambda summary, question: False)

    assert result.status == "declined"
    # result.message is exactly what commit_message produced for the changes.
    assert "1 added, 1 modified, 1 deleted." in result.message


def test_oversized_file_raises_before_confirm(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_runner, "MAX_FILE_BYTES", 5)
    report_dir = _fake_report(tmp_path / "report")
    (report_dir / "big.txt").write_text("x" * 50, encoding="utf-8")
    calls: list[str] = []

    with pytest.raises(PublishError, match="big.txt"):
        run_publish(_cfg("My Run"), report_dir, lambda s, q: calls.append(s) or True)

    assert calls == []
    assert _git(report_dir, "diff", "--cached", "--name-only", check=False).strip() == ""


def test_report_nested_in_another_repo_gets_its_own_repository(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(outer)],
        check=True, capture_output=True, text=True,
    )
    (outer / "README.md").write_text("outer repo\n", encoding="utf-8")
    _git(outer, "add", "-A")
    _git(outer, "commit", "-q", "-m", "outer baseline")
    report_dir = _fake_report(outer / "site")

    result = run_publish(_cfg("My Run"), report_dir, lambda summary, question: True)

    assert result.status == "published"
    assert result.created_repo is True
    assert (report_dir / ".git").is_dir()
    toplevel = Path(_git(report_dir, "rev-parse", "--show-toplevel").strip())
    assert toplevel.resolve() == report_dir.resolve()
    # The publish commit lives in the nested repo only.
    assert len(_git(report_dir, "log", "--format=%s").splitlines()) == 1
    assert _git(outer, "log", "--format=%s").strip() == "outer baseline"


def test_second_publish_with_upstream_uses_plain_push(tmp_path):
    bare = _bare_repo(tmp_path / "origin.git")
    report_dir = _fake_report(tmp_path / "report")
    subprocess.run(
        ["git", "init", "-b", "main", str(report_dir)],
        check=True, capture_output=True, text=True,
    )
    _git(report_dir, "remote", "add", "origin", str(bare))

    first = run_publish(_cfg("First"), report_dir, lambda summary, question: True)
    assert first.status == "published"
    assert first.push_target == "origin"  # no upstream yet: push -u origin HEAD

    (report_dir / "index.html").write_text("<html>v2</html>\n", encoding="utf-8")
    second = run_publish(_cfg("Second"), report_dir, lambda summary, question: True)

    assert second.status == "published"
    # An upstream now exists, so _push runs plain `git push` and reports it.
    assert second.push_target == "origin/main"
    assert _git(bare, "log", "--format=%s", "main").splitlines()[0] == "Update Second"
    assert int(_git(bare, "rev-list", "--count", "main").strip()) == 2


# ── CLI hooks ────────────────────────────────────────────────────────────────
def test_confirm_publish_declines_without_a_tty(monkeypatch, capsys):
    from bench.cli import _confirm_publish

    class _NoTTY:
        @staticmethod
        def isatty():
            return False

    def no_input(*args, **kwargs):
        raise AssertionError("input() must not be called on a non-TTY stdin")

    monkeypatch.setattr("bench.cli.sys.stdin", _NoTTY())
    monkeypatch.setattr("builtins.input", no_input)

    assert _confirm_publish("the summary", "Go? [y/N] ") is False
    out = capsys.readouterr().out
    assert "the summary" in out
    assert "no interactive console" in out


def test_publish_after_report_no_publish_flag_is_a_noop(tmp_path):
    from bench.cli import _publish_after_report

    report_dir = _fake_report(tmp_path / "report")

    assert _publish_after_report(_cfg(), str(report_dir), no_publish=True) == 0
    assert not (report_dir / ".git").exists()


def test_publish_after_report_disabled_publish_block_is_a_noop(tmp_path):
    from bench.cli import _publish_after_report

    report_dir = _fake_report(tmp_path / "report")
    cfg = types.SimpleNamespace(report={}, friendly_name="My Run", name="civbench-run")

    assert _publish_after_report(cfg, str(report_dir), no_publish=False) == 0
    assert not (report_dir / ".git").exists()


# ── review hardening ─────────────────────────────────────────────────────────
def test_clean_tree_offers_to_push_commits_a_failed_push_left(tmp_path):
    bare = _bare_repo(tmp_path / "origin.git")
    report_dir = _fake_report(tmp_path / "report")
    subprocess.run(
        ["git", "init", "-b", "main", str(report_dir)],
        check=True, capture_output=True, text=True,
    )
    _git(report_dir, "remote", "add", "origin", str(tmp_path / "missing.git"))

    with pytest.raises(PublishError, match="kept locally"):
        run_publish(_cfg("My Run"), report_dir, lambda summary, question: True)

    _git(report_dir, "remote", "set-url", "origin", str(bare))
    asked: list[str] = []
    result = run_publish(
        _cfg("My Run"), report_dir, lambda s, q: asked.append(s) or True
    )

    assert result.status == "pushed"
    assert result.push_target == "origin"
    assert "1 local commit(s) not yet pushed" in asked[0]
    assert _git(bare, "log", "--format=%s", "main").splitlines() == ["Update My Run"]


def test_files_hidden_by_a_global_gitignore_stop_before_prompting(tmp_path, monkeypatch):
    excludes = tmp_path / "global-ignore"
    excludes.write_text("*.csv\n", encoding="utf-8")
    config = tmp_path / "gitconfig-with-excludes"
    config.write_text(f"[core]\n\texcludesFile = {excludes.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    report_dir = _fake_report(tmp_path / "report")
    calls: list[str] = []

    with pytest.raises(PublishError, match="gitignore"):
        run_publish(_cfg("My Run"), report_dir, lambda s, q: calls.append(s) or True)

    assert calls == []


def test_missing_git_identity_stops_before_prompting(tmp_path, monkeypatch):
    for var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(var)
    config = tmp_path / "gitconfig-config-only"
    config.write_text("[user]\n\tuseConfigOnly = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    report_dir = _fake_report(tmp_path / "report")
    calls: list[str] = []

    with pytest.raises(PublishError, match="identity"):
        run_publish(_cfg("My Run"), report_dir, lambda s, q: calls.append(s) or True)

    assert calls == []
    assert _git(report_dir, "diff", "--cached", "--name-only", check=False).strip() == ""


def test_the_only_remote_is_used_when_it_is_not_origin(tmp_path):
    bare = _bare_repo(tmp_path / "pages.git")
    report_dir = _fake_report(tmp_path / "report")
    subprocess.run(
        ["git", "init", "-b", "main", str(report_dir)],
        check=True, capture_output=True, text=True,
    )
    _git(report_dir, "remote", "add", "pages", str(bare))

    result = run_publish(_cfg("My Run"), report_dir, lambda summary, question: True)

    assert result.push_target == "pages"
    assert _git(bare, "log", "--format=%s", "main").splitlines() == ["Update My Run"]


def test_confirm_publish_treats_end_of_input_as_no(monkeypatch):
    from bench.cli import _confirm_publish

    class _TTY:
        @staticmethod
        def isatty():
            return True

    def eof(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr("bench.cli.sys.stdin", _TTY())
    monkeypatch.setattr("builtins.input", eof)

    assert _confirm_publish("the summary", "Go? [y/N] ") is False


def test_publish_after_report_returns_2_on_publish_error(tmp_path, monkeypatch, capsys):
    import bench.publish
    from bench.cli import _publish_after_report

    def fail(*args, **kwargs):
        raise PublishError("boom")

    monkeypatch.setattr(bench.publish, "run_publish", fail)

    assert _publish_after_report(_cfg(), str(tmp_path), no_publish=False) == 2
    assert "publish error: boom" in capsys.readouterr().err


def test_push_names_the_upstream_when_the_local_branch_name_differs(tmp_path):
    bare = _bare_repo(tmp_path / "origin.git")
    report_dir = _fake_report(tmp_path / "report")
    subprocess.run(
        ["git", "init", "-b", "main", str(report_dir)],
        check=True, capture_output=True, text=True,
    )
    _git(report_dir, "remote", "add", "origin", str(bare))
    run_publish(_cfg("First"), report_dir, lambda summary, question: True)
    # Local branch `report` tracks origin/main; a plain `git push` would be
    # rejected under push.default=simple because the names differ.
    _git(report_dir, "branch", "-m", "main", "report")
    _git(report_dir, "config", "push.default", "simple")

    (report_dir / "index.html").write_text("<html>v2</html>\n", encoding="utf-8")
    result = run_publish(_cfg("Second"), report_dir, lambda summary, question: True)

    assert result.push_target == "origin/main"
    assert _git(bare, "log", "--format=%s", "main").splitlines() == ["Update Second", "Update First"]
    assert _git(bare, "branch", "--list", "report").strip() == ""
