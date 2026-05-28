import argparse
import pathlib
import subprocess
import unittest.mock

import pytest

import tern.main as tern_main


# ── helpers ───────────────────────────────────────────────────────────────────


def _ok(stderr: str = "", stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=stderr
    )


def _err(code: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=code, stdout="", stderr=stderr
    )


def _base_state(**kwargs) -> dict:
    state = {
        "objective": None,
        "plan": None,
        "plan_approved": None,
        "new_deps": [],
        "deps_approved": None,
        "qa_output": None,
        "issues": [],
        "need_handoff": False,
        "written_files": [],
        "feedback": [],
        "maker_checker_cycles": 0,
        "milestones": [],
        "session_objectives": [],
        "session_files": [],
    }
    state.update(kwargs)
    return state


def _snapshot(
    next_nodes: tuple = ("user",), values: dict | None = None
) -> unittest.mock.MagicMock:
    snap = unittest.mock.MagicMock()
    snap.next = next_nodes
    snap.values = values if values is not None else _base_state()
    return snap


class _AuthenticationError(Exception):
    pass


@pytest.fixture
def mock_sbx_ok():
    with unittest.mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = _ok()
        yield mock_run


@pytest.fixture
def repl_graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SANDBOX_VM_ID", "test-sandbox")
    mock_graph = unittest.mock.MagicMock()
    with unittest.mock.patch("tern.config.load_config"):
        with unittest.mock.patch("tern.agent.build_agent", return_value=mock_graph):
            yield mock_graph


# ── _sbx ──────────────────────────────────────────────────────────────────────


def test_sbx_exits_1_when_sbx_not_found():
    with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit) as exc_info:
            tern_main._sbx(["sbx", "anything"])
    assert exc_info.value.code == 1


# ── _init_scaffold ────────────────────────────────────────────────────────────


def test_init_scaffold_prints_already_exists(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    tern_main._init_scaffold(tern_dir)
    assert "already exists" in capsys.readouterr().out.lower()


def test_init_scaffold_creates_scaffold_and_prints(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    tern_dir = tmp_path / ".tern"
    with unittest.mock.patch("tern.scaffold.scaffold_and_validate") as mock_scaffold:
        tern_main._init_scaffold(tern_dir)
    mock_scaffold.assert_called_once_with(tern_dir)
    assert "initialized" in capsys.readouterr().out.lower()


def test_init_scaffold_does_not_modify_existing_dir(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    sentinel = tern_dir / "sentinel.txt"
    sentinel.write_text("original")
    tern_main._init_scaffold(tern_dir)
    assert sentinel.read_text() == "original"
    assert {f.name for f in tern_dir.iterdir()} == {"sentinel.txt"}


# ── _init_sandbox ─────────────────────────────────────────────────────────────


def test_init_sandbox_when_exists_returns_0_and_prints(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    with unittest.mock.patch(
        "subprocess.run", return_value=_ok(stdout="tern-proj running\n")
    ):
        code = tern_main._init_sandbox("tern-proj", tmp_path / ".tern")
    assert code == 0
    assert "already exists" in capsys.readouterr().out.lower()


def test_init_sandbox_when_not_exists_uses_kit_default_entrypoint(
    tmp_path: pathlib.Path,
):
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[_ok(stdout=""), _ok()],
    ) as mock_run:
        tern_main._init_sandbox(f"tern-{tmp_path.name}", tmp_path / ".tern")
    run_call = mock_run.call_args_list[1].args[0]
    assert run_call[:2] == ["sbx", "run"]
    assert "--kit" in run_call
    assert "sh" not in run_call
    assert "-c" not in run_call


def test_init_sandbox_when_not_exists_returns_sbx_exit_code(
    tmp_path: pathlib.Path,
):
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[_ok(stdout=""), _err(2)],
    ):
        code = tern_main._init_sandbox(f"tern-{tmp_path.name}", tmp_path / ".tern")
    assert code == 2


def test_init_sandbox_prints_initialized_on_success(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[_ok(stdout=""), _ok()],
    ):
        tern_main._init_sandbox(f"tern-{tmp_path.name}", tmp_path / ".tern")
    assert "initialized" in capsys.readouterr().out.lower()


# ── _remove_scaffold ──────────────────────────────────────────────────────────


def test_remove_scaffold_not_exists_prints_message(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    tern_main._remove_scaffold(tmp_path / ".tern")
    assert "not found" in capsys.readouterr().out.lower()


def test_remove_scaffold_confirm_y_removes_dir(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("builtins.input", return_value="y"):
        tern_main._remove_scaffold(tern_dir)
    assert not tern_dir.exists()


def test_remove_scaffold_confirm_y_prints_removed(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("builtins.input", return_value="y"):
        tern_main._remove_scaffold(tern_dir)
    assert "removed" in capsys.readouterr().out.lower()


def test_remove_scaffold_confirm_n_preserves_dir(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("builtins.input", return_value="n"):
        tern_main._remove_scaffold(tern_dir)
    assert tern_dir.exists()


def test_remove_scaffold_eof_preserves_dir(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("builtins.input", side_effect=EOFError):
        tern_main._remove_scaffold(tern_dir)
    assert tern_dir.exists()


def test_remove_scaffold_keyboard_interrupt_preserves_dir(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("builtins.input", side_effect=KeyboardInterrupt):
        tern_main._remove_scaffold(tern_dir)
    assert tern_dir.exists()


# ── cmd_up ────────────────────────────────────────────────────────────────────


def test_cmd_up_default_creates_scaffold(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    assert (tmp_path / ".tern").is_dir()


def test_cmd_up_default_uses_non_interactive_sandbox_init(
    tmp_path: pathlib.Path, mock_sbx_ok
):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    calls = [c.args[0] for c in mock_sbx_ok.call_args_list]
    assert ["sbx", "ls"] in calls
    assert ["sbx", "run", "--kit", str(tmp_path / ".tern"), "tern"] in calls


def test_cmd_up_default_does_not_start_repl(tmp_path: pathlib.Path, mock_sbx_ok):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    calls = [c.args[0] for c in mock_sbx_ok.call_args_list]
    assert not any("exec" in c for c in calls)


def test_cmd_up_existing_tern_dir_prints_already_exists(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture, mock_sbx_ok
):
    (tmp_path / ".tern").mkdir()
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace())
    assert "already exists" in capsys.readouterr().out.lower()


def test_cmd_up_existing_tern_dir_does_not_modify_scaffold(
    tmp_path: pathlib.Path, mock_sbx_ok
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


def test_cmd_up_exits_nonzero_when_sandbox_creation_fails(tmp_path: pathlib.Path):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch(
            "subprocess.run",
            side_effect=[
                _ok(),  # sbx kit validate (inside scaffold)
                _ok(stdout=""),  # sbx ls
                _err(3, "some error"),  # sbx run fails
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 3


def test_cmd_up_scaffold_only_calls_scaffold_no_sbx(tmp_path: pathlib.Path):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch("tern.scaffold.scaffold_and_validate"):
            with unittest.mock.patch("subprocess.run") as mock_run:
                tern_main.cmd_up(argparse.Namespace(scaffold=True, sandbox=False))
    mock_run.assert_not_called()


def test_cmd_up_scaffold_only_does_not_exit(tmp_path: pathlib.Path):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch("tern.scaffold.scaffold_and_validate"):
            tern_main.cmd_up(argparse.Namespace(scaffold=True, sandbox=False))


def test_cmd_up_sandbox_only_errors_when_scaffold_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        tern_main.cmd_up(argparse.Namespace(scaffold=False, sandbox=True))
    assert exc_info.value.code == 1


def test_cmd_up_sandbox_only_creates_sandbox_when_scaffold_present(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[_ok(stdout=""), _ok()],
    ) as mock_run:
        with pytest.raises(SystemExit):
            tern_main.cmd_up(argparse.Namespace(scaffold=False, sandbox=True))
    calls = [c.args[0] for c in mock_run.call_args_list]
    assert any("run" in c for c in calls)
    assert not any(c for c in calls if "scaffold" in str(c))


# ── cmd_on ────────────────────────────────────────────────────────────────────


def test_cmd_on_errors_when_scaffold_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        tern_main.cmd_on(argparse.Namespace())
    assert exc_info.value.code == 1


def test_cmd_on_errors_when_sandbox_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run", return_value=_ok(stdout="other running\n")
    ):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_on(argparse.Namespace())
    assert exc_info.value.code == 1


def test_cmd_on_exec_contains_repl_and_stage_message(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[
            _ok(stdout=f"tern-{tmp_path.name} running\n"),
            _ok(),
        ],
    ) as mock_run:
        with pytest.raises(SystemExit):
            tern_main.cmd_on(argparse.Namespace())
    exec_calls = [c.args[0] for c in mock_run.call_args_list if "exec" in c.args[0]]
    assert len(exec_calls) == 1
    shell_cmd = exec_calls[0][-1]
    assert "tern _repl" in shell_cmd
    assert "Preparing tern agent" in shell_cmd


def test_cmd_on_exec_uses_correct_sandbox_and_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[
            _ok(stdout=f"tern-{tmp_path.name} running\n"),
            _ok(),
        ],
    ) as mock_run:
        with pytest.raises(SystemExit):
            tern_main.cmd_on(argparse.Namespace())
    exec_call = [c.args[0] for c in mock_run.call_args_list if "exec" in c.args[0]][0]
    assert exec_call[:6] == [
        "sbx",
        "exec",
        "-it",
        "-w",
        str(tmp_path),
        f"tern-{tmp_path.name}",
    ]


def test_cmd_on_passes_through_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".tern").mkdir()
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch(
        "subprocess.run",
        side_effect=[
            _ok(stdout=f"tern-{tmp_path.name} running\n"),
            _err(2),
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_on(argparse.Namespace())
    assert exc_info.value.code == 2


# ── cmd_down ──────────────────────────────────────────────────────────────────


def test_cmd_down_default_calls_sbx_rm(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mock_sbx_ok
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        tern_main.cmd_down(argparse.Namespace())
    mock_sbx_ok.assert_called_once_with(["sbx", "rm", f"tern-{tmp_path.name}"])


def test_cmd_down_default_passes_through_sbx_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch("subprocess.run", return_value=_err(2)):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_down(argparse.Namespace())
    assert exc_info.value.code == 2


def test_cmd_down_sandbox_only_calls_sbx_rm_and_exits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mock_sbx_ok
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        tern_main.cmd_down(argparse.Namespace(scaffold=False, sandbox=True))
    mock_sbx_ok.assert_called_once_with(["sbx", "rm", f"tern-{tmp_path.name}"])


def test_cmd_down_scaffold_only_removes_dir_on_confirm(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch("builtins.input", return_value="y"):
            with unittest.mock.patch("subprocess.run") as mock_run:
                tern_main.cmd_down(argparse.Namespace(scaffold=True, sandbox=False))
    assert not tern_dir.exists()
    mock_run.assert_not_called()


def test_cmd_down_scaffold_only_preserves_on_n(tmp_path: pathlib.Path):
    tern_dir = tmp_path / ".tern"
    tern_dir.mkdir()
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch("builtins.input", return_value="n"):
            tern_main.cmd_down(argparse.Namespace(scaffold=True, sandbox=False))
    assert tern_dir.exists()


def test_cmd_down_scaffold_only_does_not_exit(tmp_path: pathlib.Path):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch("builtins.input", return_value="y"):
            tern_main.cmd_down(argparse.Namespace(scaffold=True, sandbox=False))


# ── CLI registration ──────────────────────────────────────────────────────────


def test_cli_up_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["up"]).command == "up"


def test_cli_top_level_help_shows_scaffold_and_sandbox_flags():
    parser = tern_main.get_cli_args()
    help_text = parser.format_help()
    assert "--scaffold" in help_text
    assert "--sandbox" in help_text


def test_cli_on_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["on"]).command == "on"


def test_cli_down_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["down"]).command == "down"


def test_cli_repl_subcommand_registered():
    parser = tern_main.get_cli_args()
    assert parser.parse_args(["_repl"]).command == "_repl"


def test_cli_up_scaffold_flag():
    parser = tern_main.get_cli_args()
    args = parser.parse_args(["up", "--scaffold"])
    assert args.scaffold is True
    assert args.sandbox is False


def test_cli_up_sandbox_flag():
    parser = tern_main.get_cli_args()
    args = parser.parse_args(["up", "--sandbox"])
    assert args.sandbox is True
    assert args.scaffold is False


def test_cli_down_scaffold_flag():
    parser = tern_main.get_cli_args()
    args = parser.parse_args(["down", "--scaffold"])
    assert args.scaffold is True


def test_cli_down_sandbox_flag():
    parser = tern_main.get_cli_args()
    args = parser.parse_args(["down", "--sandbox"])
    assert args.sandbox is True


# ── _is_auth_error ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_AuthenticationError("bad key"), True),
        (ValueError("authentication failed"), True),
        (ValueError("invalid api key provided"), True),
        (ValueError("401 unauthorized"), True),
        (ValueError("something went wrong"), False),
    ],
    ids=["class_name", "auth_msg", "api_key_msg", "unauthorized_msg", "generic"],
)
def test_is_auth_error(exc: Exception, expected: bool):
    assert tern_main._is_auth_error(exc) == expected


# ── sandbox_exists ────────────────────────────────────────────────────────────


def test_sandbox_exists_false_when_missing():
    with unittest.mock.patch(
        "subprocess.run", return_value=_ok(stdout="other running\n")
    ):
        assert tern_main._sandbox_exists("tern-project") is False


def test_sandbox_exists_true_when_name_matches():
    with unittest.mock.patch(
        "subprocess.run", return_value=_ok(stdout="tern-project running\n")
    ):
        assert tern_main._sandbox_exists("tern-project") is True


# ── _detect_checkpoint ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "state_kwargs, expected",
    [
        ({}, "new_objective"),
        ({"plan": "step 1", "plan_approved": None}, "plan_approval"),
        ({"new_deps": ["numpy"], "deps_approved": None}, "dep_approval"),
        ({"issues": ["bad import"], "plan_approved": None}, "new_objective"),
        ({"qa_output": "all passed", "plan_approved": True}, "new_objective"),
        (
            {
                "new_deps": ["numpy"],
                "deps_approved": None,
                "plan": "step 1",
                "plan_approved": None,
            },
            "dep_approval",
        ),
        (
            {"issues": ["bad"], "qa_output": "output", "plan_approved": None},
            "new_objective",
        ),
        (
            {"issues": ["bad import"], "plan": "step 1", "plan_approved": None},
            "new_objective",
        ),
        ({"plan": "step 1", "plan_approved": False}, "plan_approval"),
    ],
    ids=[
        "empty",
        "plan_approval",
        "dep_approval",
        "new_objective_with_issues",
        "cycle_complete_maps_to_new_objective",
        "dep_beats_plan",
        "issues_map_to_new_objective",
        "issues_beats_plan",
        "dep_rejected_routes_to_plan_approval",
    ],
)
def test_detect_checkpoint(state_kwargs: dict, expected: str):
    assert tern_main._detect_checkpoint(_base_state(**state_kwargs)) == expected


# ── _compute_update ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "checkpoint, user_input, expected",
    [
        ("plan_approval", "approve", {"plan_approved": True, "feedback": []}),
        ("plan_approval", "fix the imports", {"feedback": ["fix the imports"]}),
        ("dep_approval", "approve", {"deps_approved": True, "feedback": []}),
        (
            "dep_approval",
            "use stdlib instead",
            {
                "deps_approved": None,
                "plan_approved": False,
                "new_deps": [],
                "issues": [],
                "feedback": ["use stdlib instead"],
            },
        ),
    ],
    ids=[
        "plan_approve",
        "plan_feedback",
        "dep_approve",
        "dep_reject",
    ],
)
def test_compute_update_approvals(checkpoint: str, user_input: str, expected: dict):
    assert tern_main._compute_update(checkpoint, user_input) == expected


def test_compute_update_objective_clears_stale():
    result = tern_main._compute_update("new_objective", "build a classifier")
    assert result["objective"] == "build a classifier"
    assert result["qa_output"] is None
    assert result["plan_approved"] is None
    assert result["deps_approved"] is None
    assert result["feedback"] == []
    assert result["new_deps"] == []
    assert result["maker_checker_cycles"] == 0
    assert result["session_objectives"] == ["build a classifier"]
    assert "issues" not in result


# ── _invoke ───────────────────────────────────────────────────────────────────


def test_invoke_returns_true_on_success():
    mock_graph = unittest.mock.MagicMock()
    assert tern_main._invoke(mock_graph, None, {}) is True


def test_invoke_returns_false_and_prints_on_auth_error(
    capsys: pytest.CaptureFixture,
):
    mock_graph = unittest.mock.MagicMock()
    mock_graph.invoke.side_effect = _AuthenticationError("bad key")
    result = tern_main._invoke(mock_graph, None, {})
    assert result is False
    assert "API authentication failed" in capsys.readouterr().err


def test_invoke_re_raises_non_auth_error():
    mock_graph = unittest.mock.MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("network error")
    with pytest.raises(RuntimeError):
        tern_main._invoke(mock_graph, None, {})


# ── _print_checkpoint ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "checkpoint, state_kwargs, needle",
    [
        ("plan_approval", {"plan": "step 1: build model"}, "step 1: build model"),
        ("dep_approval", {"new_deps": ["numpy"]}, "numpy"),
        (
            "new_objective",
            {"issues": ["bad import"], "plan_approved": None},
            "bad import",
        ),
        ("new_objective", {"qa_output": "all passed"}, "Milestone complete"),
        ("new_objective", {}, None),
    ],
    ids=[
        "plan",
        "dep",
        "new_objective_with_issues",
        "cycle_complete_via_new_objective",
        "new_objective_blank",
    ],
)
def test_print_checkpoint(
    checkpoint: str,
    state_kwargs: dict,
    needle: str | None,
    capsys: pytest.CaptureFixture,
):
    tern_main._print_checkpoint(checkpoint, _base_state(**state_kwargs))
    out = capsys.readouterr().out
    if needle is not None:
        assert needle in out
    else:
        assert out == ""


def test_print_checkpoint_new_objective_after_cycle_shows_written_files(
    capsys: pytest.CaptureFixture,
):
    tern_main._print_checkpoint(
        "new_objective",
        _base_state(qa_output="ok", written_files=["temp/hello.py", "src/foo.py"]),
    )
    out = capsys.readouterr().out
    assert "temp/hello.py" in out
    assert "src/foo.py" in out


def test_print_checkpoint_new_objective_after_cycle_no_written_files(
    capsys: pytest.CaptureFixture,
):
    tern_main._print_checkpoint(
        "new_objective", _base_state(qa_output="ok", written_files=[])
    )
    out = capsys.readouterr().out
    assert "Milestone complete" in out
    assert "Files written" not in out


# ── _prompt ───────────────────────────────────────────────────────────────────


def test_prompt_returns_stripped_input():
    with unittest.mock.patch("builtins.input", return_value="  approve  "):
        assert tern_main._prompt("plan_approval") == "approve"


def test_prompt_returns_none_on_keyboard_interrupt():
    with unittest.mock.patch("builtins.input", side_effect=KeyboardInterrupt):
        assert tern_main._prompt("plan_approval") is None


def test_prompt_returns_none_on_eof():
    with unittest.mock.patch("builtins.input", side_effect=EOFError):
        assert tern_main._prompt("new_objective") is None


# ── cmd_repl ──────────────────────────────────────────────────────────────────


def test_cmd_repl_exits_1_when_sandbox_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SANDBOX_VM_ID", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        tern_main.cmd_repl(argparse.Namespace())
    assert exc_info.value.code == 1


def test_cmd_repl_exits_immediately_when_graph_at_end(repl_graph):
    repl_graph.get_state.return_value = _snapshot(next_nodes=())
    with unittest.mock.patch("builtins.input") as mock_input:
        tern_main.cmd_repl(argparse.Namespace())
    mock_input.assert_not_called()


def test_cmd_repl_exit_no_handoff_does_not_update_state(repl_graph):
    repl_graph.get_state.return_value = _snapshot(values=_base_state(objective=None))
    with unittest.mock.patch("builtins.input", side_effect=["exit", "n"]):
        tern_main.cmd_repl(argparse.Namespace())
    repl_graph.update_state.assert_not_called()


def test_cmd_repl_exit_with_handoff_and_objective_sets_need_handoff(repl_graph):
    repl_graph.get_state.return_value = _snapshot(
        values=_base_state(objective="build a model")
    )
    with unittest.mock.patch("builtins.input", side_effect=["exit", "y"]):
        tern_main.cmd_repl(argparse.Namespace())
    repl_graph.update_state.assert_called_once_with(
        unittest.mock.ANY, {"need_handoff": True}
    )


def test_cmd_repl_exit_with_handoff_no_objective_does_not_invoke_graph(repl_graph):
    repl_graph.get_state.return_value = _snapshot(values=_base_state(objective=None))
    with unittest.mock.patch("builtins.input", side_effect=["exit", "y"]):
        tern_main.cmd_repl(argparse.Namespace())
    assert repl_graph.invoke.call_count == 1  # only the initial invoke


def test_cmd_repl_keyboard_interrupt_exits_loop(repl_graph):
    repl_graph.get_state.return_value = _snapshot(values=_base_state())
    with unittest.mock.patch("builtins.input", side_effect=[KeyboardInterrupt, "n"]):
        tern_main.cmd_repl(argparse.Namespace())


def test_cmd_repl_plan_approval_approve_calls_correct_update(repl_graph):
    plan_state = _base_state(plan="step 1: build model", plan_approved=None)
    repl_graph.get_state.side_effect = [
        _snapshot(values=plan_state),
        _snapshot(next_nodes=()),
    ]
    with unittest.mock.patch("builtins.input", return_value="approve"):
        tern_main.cmd_repl(argparse.Namespace())
    repl_graph.update_state.assert_called_once_with(
        unittest.mock.ANY, {"plan_approved": True, "feedback": []}
    )


def test_cmd_repl_auth_error_during_invoke_exits_1(
    repl_graph, capsys: pytest.CaptureFixture
):
    repl_graph.get_state.return_value = _snapshot(
        values=_base_state(plan="step 1", plan_approved=None)
    )
    repl_graph.invoke.side_effect = [None, _AuthenticationError("invalid api key")]
    with unittest.mock.patch("builtins.input", return_value="approve"):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_repl(argparse.Namespace())
    assert exc_info.value.code == 1
    assert "API authentication failed" in capsys.readouterr().err


def test_cmd_repl_auth_error_during_handoff_returns_cleanly(
    repl_graph, capsys: pytest.CaptureFixture
):
    repl_graph.get_state.return_value = _snapshot(
        values=_base_state(objective="build thing")
    )
    repl_graph.invoke.side_effect = [None, _AuthenticationError("bad key")]
    with unittest.mock.patch("builtins.input", side_effect=["exit", "y"]):
        tern_main.cmd_repl(argparse.Namespace())
    assert "API authentication failed" in capsys.readouterr().err
