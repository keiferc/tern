import argparse
import pathlib
import subprocess
import unittest.mock

import pytest

import tern.main as tern_main


@pytest.fixture
def mock_sbx_ok():
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        yield mock_run


# ── cmd_init ──────────────────────────────────────────────────────────────────


def test_cmd_init_creates_tern_dir(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        tern_main.cmd_init(argparse.Namespace())
    assert (tmp_path / ".tern").is_dir()


def test_cmd_init_existing_tern_dir_is_noop(
    tmp_path: pathlib.Path, capsys, mock_sbx_ok
):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    sentinel = tern_dir / "sentinel.txt"
    sentinel.write_text("original")
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        tern_main.cmd_init(argparse.Namespace())
    assert sentinel.read_text() == "original"
    assert {f.name for f in tern_dir.iterdir()} == {"sentinel.txt"}
    assert "warning" in capsys.readouterr().out.lower()


# ── cmd_up ────────────────────────────────────────────────────────────────────


def test_cmd_up_exits_1(capsys):
    with pytest.raises(SystemExit) as exc_info:
        tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 1


# ── CLI registration ──────────────────────────────────────────────────────────


def test_cli_init_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["init"]).command == "init"


def test_cli_up_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["up"]).command == "up"
