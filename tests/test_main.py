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
    assert ["sbx", "ls"] in calls
    assert [
        "sbx",
        "run",
        "--kit",
        str(tmp_path / ".tern"),
        "--name",
        f"tern-{tmp_path.name}",
        "tern",
        str(tmp_path),
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


def test_cmd_up_exits_nonzero_when_sandbox_creation_fails(tmp_path: pathlib.Path):
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        with unittest.mock.patch(
            "subprocess.run",
            side_effect=[
                _ok(),  # sbx kit validate (inside scaffold)
                _ok(stdout=""),  # sbx ls
                _err(3, "some error"),  # sbx run --kit fails
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 3


def test_cmd_up_existing_sandbox_uses_exec(
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
            tern_main.cmd_up(argparse.Namespace())
    calls = [c.args[0] for c in mock_run.call_args_list]
    assert [
        "sbx",
        "exec",
        "-it",
        "-w",
        str(tmp_path),
        f"tern-{tmp_path.name}",
        "sh",
        "-lc",
        "uv sync --quiet && /home/agent/.venv/bin/tern _repl",
    ] in calls


def test_cmd_up_existing_sandbox_passes_through_repl_exit_code(
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
            tern_main.cmd_up(argparse.Namespace())
    assert exc_info.value.code == 2


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


# ── cmd_down ──────────────────────────────────────────────────────────────────


def test_cmd_down_calls_sbx_rm_with_sandbox_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mock_sbx_ok
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        tern_main.cmd_down(argparse.Namespace())
    mock_sbx_ok.assert_called_once_with(["sbx", "rm", f"tern-{tmp_path.name}"])


def test_cmd_down_passes_through_sbx_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with unittest.mock.patch("subprocess.run", return_value=_err(2)):
        with pytest.raises(SystemExit) as exc_info:
            tern_main.cmd_down(argparse.Namespace())
    assert exc_info.value.code == 2


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
                "plan_approved": None,
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
        _snapshot(values=plan_state),  # loop iteration 1
        _snapshot(next_nodes=()),  # loop iteration 2 — exits
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
