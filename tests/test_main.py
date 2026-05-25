import argparse
import pathlib
import subprocess
import unittest.mock

import pytest

import tern.main as tern_main


def _ok(stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=stderr)


def _err(code: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=code, stdout="", stderr=stderr
    )


@pytest.fixture
def mock_sbx_ok():
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = _ok()
        yield mock_run


# ── _sbx ──────────────────────────────────────────────────────────────────────


def test_sbx_exits_1_when_sbx_not_found():
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit) as exc_info:
            tern_main._sbx(["sbx", "anything"])
    assert exc_info.value.code == 1


# ── cmd_up ────────────────────────────────────────────────────────────────────


def test_cmd_up_creates_tern_dir(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    assert (tmp_path / ".tern").is_dir()


def test_cmd_up_creates_sandbox(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    calls = [c.args[0] for c in mock_sbx_ok.call_args_list]
    assert ["sbx", "create", "shell", ".", "--kit", "./.tern/"] in calls


def test_cmd_up_syncs_venv_before_repl(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    calls = [c.args[0] for c in mock_sbx_ok.call_args_list]
    assert [
        "sbx",
        "exec",
        "-w",
        str(tmp_path),
        f"shell-{tmp_path.name}",
        "env",
        "UV_PROJECT_ENVIRONMENT=/home/agent/.venv",
        "uv",
        "sync",
    ] in calls


def test_cmd_up_execs_repl(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    calls = [c.args[0] for c in mock_sbx_ok.call_args_list]
    assert [
        "sbx",
        "exec",
        "-w",
        str(tmp_path),
        f"shell-{tmp_path.name}",
        "/home/agent/.venv/bin/tern",
        "_repl",
    ] in calls


def test_cmd_up_existing_tern_dir_skips_scaffold(
    tmp_path: pathlib.Path, capsys, mock_sbx_ok
):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    sentinel = tern_dir / "sentinel.txt"
    sentinel.write_text("original")
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    assert sentinel.read_text() == "original"
    assert {f.name for f in tern_dir.iterdir()} == {"sentinel.txt"}
    assert "skipping scaffold" in capsys.readouterr().out.lower()


def test_cmd_up_skips_create_when_sandbox_already_exists(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[
            _err(1, "sandbox already exists"),
            _ok(),
            _ok(),
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 0


def test_cmd_up_exits_nonzero_when_sandbox_creation_fails(tmp_path: pathlib.Path):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch(
            "subprocess.run",
            side_effect=[
                _ok(),
                _err(3, "some error"),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 3


def test_cmd_up_exits_nonzero_when_sync_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run", side_effect=[_ok(), _err(5, "sync error")]
    ):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 5


def test_cmd_up_passes_through_repl_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch("subprocess.run", side_effect=[_ok(), _ok(), _err(2)]):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 2


# ── cmd_down ──────────────────────────────────────────────────────────────────


def test_cmd_down_calls_sbx_rm_with_sandbox_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mock_sbx_ok
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        tern_main.cmd_down(argparse.Namespace())
    mock_sbx_ok.assert_called_once_with(["sbx", "rm", f"shell-{tmp_path.name}"])


def test_cmd_down_passes_through_sbx_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch("subprocess.run", return_value=_err(2)):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_down(argparse.Namespace())
    assert exc_info.value.code == 2


# ── cmd_repl ──────────────────────────────────────────────────────────────────


def test_cmd_repl_exits_1_when_sandbox_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SANDBOX_VM_ID", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        tern_main.cmd_repl(argparse.Namespace())
    assert exc_info.value.code == 1


# ── CLI registration ──────────────────────────────────────────────────────────


def test_cli_up_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["up"]).command == "up"


def test_cli_down_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["down"]).command == "down"


def test_cli_repl_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["_repl"]).command == "_repl"
