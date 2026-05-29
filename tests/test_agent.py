import operator
import pathlib
import typing as T
import unittest.mock

import langgraph.graph as lg_graph
import pytest

import tern.agent as agent
import tern.config as tern_config
import tern.subagents as tern_subagents
import tern.ui as tern_ui


def make_config() -> tern_config.Config:
    return tern_config.Config(
        models={
            a: "anthropic:claude-sonnet-4-6"
            for a in ("planner", "maker", "checker", "summarizer")
        },
        checker_tools=[],
        max_iterations={
            a: 20
            for a in (
                "planner",
                "maker",
                "checker",
                "summarizer",
                "maker_checker_cycles",
            )
        },
    )


def make_state(**kwargs: T.Any) -> agent.AgentState:
    return T.cast(
        agent.AgentState,
        {
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
            **kwargs,
        },
    )


# ── graph ─────────────────────────────────────────────────────────────────


def test_graph_compiles(tmp_path: pathlib.Path):
    assert agent.build_agent(make_config(), tmp_path) is not None


# ── route_from_user ───────────────────────────────────────────────────────


def test_route_user_no_objective_no_handoff():
    state = make_state(objective=None, need_handoff=False)
    assert agent.route_from_user(state) == lg_graph.END


def test_route_user_no_objective_with_handoff_routes_to_end():
    state = make_state(objective=None, need_handoff=True)
    assert agent.route_from_user(state) == lg_graph.END


def test_route_user_with_objective_and_handoff_routes_to_summarizer():
    state = make_state(objective="build a model", need_handoff=True)
    assert agent.route_from_user(state) == "summarizer"


def test_route_user_cycle_complete():
    state = make_state(
        objective="build a model",
        plan_approved=True,
        qa_output="all tests passed",
        issues=[],
    )
    assert agent.route_from_user(state) == "user"


def test_route_user_stale_qa_output_plan_not_approved():
    # plan check must fire before cycle-complete check: stale qa_output from a
    # previous cycle must not prevent routing to planner.
    state = make_state(
        objective="build a model",
        plan_approved=False,
        qa_output="all tests passed",
        issues=[],
    )
    assert agent.route_from_user(state) == "planner"


def test_route_user_plan_not_yet_approved():
    state = make_state(objective="build a model", plan_approved=None)
    assert agent.route_from_user(state) == "planner"


def test_route_user_plan_rejected():
    state = make_state(objective="build a model", plan_approved=False)
    assert agent.route_from_user(state) == "planner"


def test_route_user_deps_approved():
    state = make_state(
        objective="build a model",
        plan_approved=True,
        new_deps=["pandas"],
        deps_approved=True,
    )
    assert agent.route_from_user(state) == "qa_runner"


def test_route_user_deps_rejected():
    state = make_state(
        objective="build a model",
        plan_approved=True,
        new_deps=["pandas"],
        deps_approved=False,
    )
    assert agent.route_from_user(state) == "maker"


def test_route_user_stale_deps_approved_routes_to_maker():
    state = make_state(
        objective="build a model",
        plan_approved=True,
        new_deps=[],
        deps_approved=True,
        qa_output=None,
    )
    assert agent.route_from_user(state) == "maker"


def test_route_user_plan_approved_no_deps():
    state = make_state(
        objective="build a model",
        plan_approved=True,
        new_deps=[],
        deps_approved=None,
    )
    assert agent.route_from_user(state) == "maker"


# ── route_from_dep_check ──────────────────────────────────────────────────


def test_route_dep_check_new_deps():
    state = make_state(new_deps=["pandas", "numpy"])
    assert agent.route_from_dep_check(state) == "user"


def test_route_dep_check_no_new_deps():
    state = make_state(new_deps=[])
    assert agent.route_from_dep_check(state) == "qa_runner"


# ── route_from_checker ────────────────────────────────────────────────────


def test_route_checker_has_issues():
    state = make_state(issues=["unused import on line 3"], plan_approved=True)
    assert agent.route_from_checker(state) == "maker"


def test_route_checker_no_issues():
    state = make_state(issues=[])
    assert agent.route_from_checker(state) == "user"


def test_route_checker_issues_at_cycle_limit_routes_to_user():
    state = make_state(issues=["unused import on line 3"], plan_approved=None)
    assert agent.route_from_checker(state) == "user"


# ── nodes ─────────────────────────────────────────────────────────────────


def test_user_node_returns_empty_dict():
    state = make_state()
    assert agent.user_node(state) == {}


def test_planner_node_prints_planning_banner(tmp_path: pathlib.Path):
    state = make_state(objective="build a model")
    with unittest.mock.patch("tern.subagents.planner_subagent", return_value="plan"):
        with unittest.mock.patch.object(tern_ui, "print_stage") as mock_stage:
            agent.planner_node(state, make_config(), tmp_path)
    mock_stage.assert_called_once_with("Planning")


def test_maker_node_prints_implementing_banner(tmp_path: pathlib.Path):
    state = make_state(objective="build a model", plan="step 1")
    with unittest.mock.patch.object(tern_subagents, "maker_subagent", return_value=[]):
        with unittest.mock.patch.object(tern_ui, "print_stage") as mock_stage:
            agent.maker_node(state, make_config(), tmp_path)
    mock_stage.assert_called_once_with("Implementing")


def test_dep_check_node_prints_reviewing_banner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    with unittest.mock.patch.object(tern_ui, "print_stage") as mock_stage:
        agent.dep_check_node(make_state())
    mock_stage.assert_called_once_with("Reviewing")


def test_qa_runner_node_does_not_print_stage_banner():
    with unittest.mock.patch.object(tern_ui, "print_stage") as mock_stage:
        agent.qa_runner_node(make_config())
    mock_stage.assert_not_called()


def test_checker_node_does_not_print_stage_banner(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=["ghost.py"])
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        with unittest.mock.patch.object(tern_ui, "print_stage") as mock_stage:
            agent.checker_node(state, make_config(), tmp_path)
    mock_stage.assert_not_called()


def test_summarizer_node_prints_generating_handoff_banner(tmp_path: pathlib.Path):
    state = make_state()
    with unittest.mock.patch.object(
        tern_subagents, "summarizer_subagent", return_value=""
    ):
        with unittest.mock.patch.object(tern_ui, "print_stage") as mock_stage:
            agent.summarizer_node(state, make_config(), tmp_path)
    mock_stage.assert_called_once_with("Generating handoff")


def test_checker_node_prints_done_no_issues(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    state = make_state(qa_output="", written_files=["ghost.py"])
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        agent.checker_node(state, make_config(), tmp_path)
    assert "done — no issues" in capsys.readouterr().out


def test_checker_node_prints_issues(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    state = make_state(qa_output="", written_files=["ghost.py"])
    with unittest.mock.patch.object(
        tern_subagents,
        "checker_subagent",
        return_value=["unused import", "missing type hint"],
    ):
        agent.checker_node(state, make_config(), tmp_path)
    out = capsys.readouterr().out
    assert "  - unused import" in out
    assert "  - missing type hint" in out


def test_checker_node_prints_synthetic_issue_when_no_files(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
):
    state = make_state(qa_output="", written_files=[], maker_checker_cycles=1)
    agent.checker_node(state, make_config(), tmp_path)
    assert "  - Maker wrote no files" in capsys.readouterr().out


def test_planner_node_returns_plan_fields(tmp_path: pathlib.Path):
    state = make_state(objective="build a model")
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="step 1: do thing"
    ):
        result = agent.planner_node(state, make_config(), tmp_path)
    assert result["plan"] == "step 1: do thing"
    assert result["plan_approved"] is None
    assert result["issues"] == []
    assert result["feedback"] == []
    assert result["maker_checker_cycles"] == 0


def test_planner_node_clears_qa_output(tmp_path: pathlib.Path):
    state = make_state(objective="build a model", qa_output="prior output")
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="new plan"
    ):
        result = agent.planner_node(state, make_config(), tmp_path)
    assert result["qa_output"] is None


def test_planner_node_clears_issues(tmp_path: pathlib.Path):
    state = make_state(
        objective="build a model", issues=["stale issue from prior cycle"]
    )
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="new plan"
    ):
        result = agent.planner_node(state, make_config(), tmp_path)
    assert result["issues"] == []


def test_planner_node_passes_context_to_planner_subagent(tmp_path: pathlib.Path):
    state = make_state(
        objective="build a model",
        plan="old plan",
        issues=["unused import"],
        feedback=["fix imports"],
    )
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="new plan"
    ) as mock_planner:
        agent.planner_node(state, make_config(), tmp_path)
    kwargs = mock_planner.call_args.kwargs
    assert kwargs.get("prior_plan") == "old plan"
    assert kwargs.get("issues") == ["unused import"]
    assert kwargs.get("feedback") == ["fix imports"]


def test_planner_node_passes_handoff_when_file_exists(tmp_path: pathlib.Path):
    (tmp_path / "HANDOFF.md").write_text("prior session content", encoding="utf-8")
    state = make_state(objective="build a model")
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="plan"
    ) as mock_planner:
        agent.planner_node(state, make_config(), tmp_path)
    assert mock_planner.call_args.kwargs.get("handoff") == "prior session content"


def test_planner_node_omits_handoff_when_file_absent(tmp_path: pathlib.Path):
    state = make_state(objective="build a model")
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="plan"
    ) as mock_planner:
        agent.planner_node(state, make_config(), tmp_path)
    assert mock_planner.call_args.kwargs.get("handoff") is None


def test_maker_node_passes_context_to_maker_subagent(tmp_path: pathlib.Path):
    state = make_state(
        objective="build a model",
        plan="step 1",
        issues=["unused import"],
        feedback=["add type hints"],
    )
    with unittest.mock.patch.object(
        tern_subagents, "maker_subagent", return_value=[]
    ) as mock_maker:
        agent.maker_node(state, make_config(), tmp_path)
    assert mock_maker.call_args.args[0] == "build a model"
    assert mock_maker.call_args.kwargs.get("issues") == ["unused import"]
    assert mock_maker.call_args.kwargs.get("feedback") == ["add type hints"]


def test_maker_node_increments_maker_checker_cycles(tmp_path: pathlib.Path):
    state = make_state(plan="step 1: do thing", maker_checker_cycles=1)
    with unittest.mock.patch.object(tern_subagents, "maker_subagent", return_value=[]):
        result = agent.maker_node(state, make_config(), tmp_path)
    assert result["maker_checker_cycles"] == 2


def test_maker_node_returns_written_files(tmp_path: pathlib.Path):
    state = make_state(plan="step 1: do thing")
    with unittest.mock.patch.object(
        tern_subagents, "maker_subagent", return_value=["foo.py"]
    ):
        result = agent.maker_node(state, make_config(), tmp_path)
    assert result == {"written_files": ["foo.py"], "maker_checker_cycles": 1}


def test_qa_runner_node_empty_tools():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    assert agent.qa_runner_node(config) == {"qa_output": ""}


def test_qa_runner_node_captures_stdout():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=["echo hello"],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    result = agent.qa_runner_node(config)
    assert "hello" in result["qa_output"]


def test_qa_runner_node_captures_stderr():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=["python3 -c \"import sys; sys.stderr.write('err\\n')\""],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    result = agent.qa_runner_node(config)
    assert "err" in result["qa_output"]


def test_qa_runner_node_no_raise_on_nonzero_exit():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=['python3 -c "import sys; sys.exit(1)"'],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    result = agent.qa_runner_node(config)
    assert "qa_output" in result


def test_qa_runner_node_no_raise_on_command_not_found():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=["no_such_binary_xyz"],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    result = agent.qa_runner_node(config)
    assert "Error:" in result["qa_output"]


def test_qa_runner_node_skips_empty_cmd():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=["", "echo ok"],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    result = agent.qa_runner_node(config)
    assert "ok" in result["qa_output"]


def test_qa_runner_node_labels_each_command():
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=["echo first", "echo second"],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )
    result = agent.qa_runner_node(config)
    assert "$ echo first" in result["qa_output"]
    assert "$ echo second" in result["qa_output"]


def test_checker_node_wraps_subagent_result(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=["ghost.py"])
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["unused import on line 3"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result == {"issues": ["unused import on line 3"], "feedback": []}


def test_checker_node_formats_written_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("x = 1")
    state = make_state(qa_output="", written_files=[str(tmp_path / "foo.py")])
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ) as mock_checker:
        agent.checker_node(state, make_config(), tmp_path)
    file_contents_arg = mock_checker.call_args[0][1]
    assert "foo.py" in file_contents_arg
    assert "x = 1" in file_contents_arg


def test_checker_node_skips_missing_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    state = make_state(
        qa_output="", written_files=[str(tmp_path / "ghost.py")], plan="step 1"
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result == {
        "issues": [],
        "feedback": [],
        "maker_checker_cycles": 0,
        "milestones": ["step 1"],
        "session_files": [str(tmp_path / "ghost.py")],
    }


def test_checker_node_silently_skips_path_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.py"
    state = make_state(qa_output="", written_files=[str(outside)], plan="step 1")
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result == {
        "issues": [],
        "feedback": [],
        "maker_checker_cycles": 0,
        "milestones": ["step 1"],
        "session_files": [str(outside)],
    }


def test_summarizer_node_returns_empty_dict(tmp_path: pathlib.Path):
    state = make_state()
    assert agent.summarizer_node(state, make_config(), tmp_path) == {}


def test_summarizer_node_writes_handoff_doc(tmp_path: pathlib.Path):
    state = make_state()
    with unittest.mock.patch.object(
        tern_subagents, "summarizer_subagent", return_value="# Handoff\n\nDone."
    ):
        agent.summarizer_node(state, make_config(), tmp_path)
    assert (tmp_path / "HANDOFF.md").read_text() == "# Handoff\n\nDone."


def test_summarizer_node_skips_write_when_empty(tmp_path: pathlib.Path):
    state = make_state()
    agent.summarizer_node(state, make_config(), tmp_path)
    assert not (tmp_path / "HANDOFF.md").exists()


# ── feedback field ────────────────────────────────────────────────────────


def test_agent_state_feedback_field_has_no_reducer():
    annotated_args = T.get_args(agent.AgentState.__annotations__["feedback"])
    assert operator.add not in annotated_args


def test_planner_node_clears_feedback_and_resets_cycles(tmp_path: pathlib.Path):
    state = make_state(
        objective="build a model", feedback=["fix imports"], maker_checker_cycles=2
    )
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="new plan"
    ):
        result = agent.planner_node(state, make_config(), tmp_path)
    assert result["feedback"] == []
    assert result["maker_checker_cycles"] == 0


def test_checker_node_appends_checkpoint_on_clean_pass(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="", written_files=["ghost.py"], plan="step 1: build model"
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result["milestones"] == ["step 1: build model"]


def test_checker_node_does_not_append_checkpoint_on_issues(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="", written_files=["ghost.py"], plan="step 1: build model"
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["unused import"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert "milestones" not in result


def test_checker_node_resets_cycles_on_clean_pass(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=["ghost.py"], maker_checker_cycles=2)
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result["maker_checker_cycles"] == 0


def test_checker_node_resets_plan_approved_at_cycle_limit(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="",
        written_files=["ghost.py"],
        plan_approved=True,
        maker_checker_cycles=3,
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["issue one"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result.get("plan_approved") is None


def test_checker_node_does_not_reset_plan_approved_below_limit(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="",
        written_files=["ghost.py"],
        plan_approved=True,
        maker_checker_cycles=2,
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["issue one"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert "plan_approved" not in result


def test_checker_node_clears_feedback_when_no_issues(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="", written_files=["ghost.py"], feedback=["fix imports"]
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result.get("feedback") == []


def test_checker_node_clears_feedback_when_issues_found(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="", written_files=["ghost.py"], feedback=["fix imports"]
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["unused import on line 3"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result.get("feedback") == []


def test_checker_node_clears_feedback_at_cycle_limit(tmp_path: pathlib.Path):
    state = make_state(
        qa_output="",
        written_files=["ghost.py"],
        feedback=["fix imports"],
        plan_approved=True,
        maker_checker_cycles=3,
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["unused import on line 3"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result.get("feedback") == []


def test_checker_node_no_written_files_returns_synthetic_issue(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=[], maker_checker_cycles=1)
    result = agent.checker_node(state, make_config(), tmp_path)
    assert result["issues"] == [
        "Maker wrote no files. Use write_file to implement the plan."
    ]
    assert "plan_approved" not in result


def test_checker_node_no_written_files_at_cycle_limit(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=[], maker_checker_cycles=20)
    result = agent.checker_node(state, make_config(), tmp_path)
    assert result["issues"] == [
        "Maker wrote no files. Use write_file to implement the plan."
    ]
    assert result.get("plan_approved") is None


def test_checker_node_passes_plan_and_feedback_to_checker_subagent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("x = 1")
    state = make_state(
        qa_output="",
        written_files=[str(tmp_path / "foo.py")],
        plan="step 1: build model",
        feedback=["ignore coverage warnings"],
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ) as mock_checker:
        agent.checker_node(state, make_config(), tmp_path)
    assert mock_checker.call_args.kwargs.get("plan") == "step 1: build model"
    assert mock_checker.call_args.kwargs.get("feedback") == ["ignore coverage warnings"]


def test_checker_node_appends_session_files_on_clean_pass(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("x = 1")
    state = make_state(
        qa_output="",
        written_files=[str(tmp_path / "foo.py")],
        plan="step 1",
    )
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert result["session_files"] == [str(tmp_path / "foo.py")]


def test_checker_node_does_not_append_session_files_on_issues(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=["ghost.py"])
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=["unused import"]
    ):
        result = agent.checker_node(state, make_config(), tmp_path)
    assert "session_files" not in result


# ── messages accumulation ─────────────────────────────────────────────────


def test_milestones_field_uses_add_reducer():
    annotated_args = T.get_args(agent.AgentState.__annotations__["milestones"])
    assert operator.add in annotated_args


def test_session_objectives_field_uses_add_reducer():
    annotated_args = T.get_args(agent.AgentState.__annotations__["session_objectives"])
    assert operator.add in annotated_args


def test_session_files_field_uses_add_reducer():
    annotated_args = T.get_args(agent.AgentState.__annotations__["session_files"])
    assert operator.add in annotated_args


# ── dep_check_node ────────────────────────────────────────────────────────


def _write_dep_files(
    tmp_path: pathlib.Path,
    deps: list[str],
    lock_packages: list[str],
) -> None:
    deps_toml = "\n".join(f'  "{d}",' for d in deps)
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\ndependencies = [\n{deps_toml}\n]\n", encoding="utf-8"
    )
    pkg_entries = "\n\n".join(
        f'[[package]]\nname = "{p}"\nversion = "1.0.0"' for p in lock_packages
    )
    (tmp_path / "uv.lock").write_text(
        f"version = 1\n\n{pkg_entries}\n", encoding="utf-8"
    )


def test_dep_check_node_in_sync(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_dep_files(tmp_path, ["pandas>=1.0"], ["pandas"])
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": []}


def test_dep_check_node_returns_missing_package(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_dep_files(tmp_path, ["pandas>=1.0", "numpy"], ["pandas"])
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": ["numpy"]}


def test_dep_check_node_strips_version_specifier(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_dep_files(tmp_path, ["pandas>=1.0,<2.0"], ["pandas"])
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": []}


def test_dep_check_node_normalizes_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_dep_files(tmp_path, ["my_pkg"], ["my-pkg"])
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": []}


def test_dep_check_node_no_project_section(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": []}


def test_dep_check_node_no_dependencies_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'foo'\n", encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": []}


def test_dep_check_node_empty_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_dep_files(tmp_path, ["pandas"], [])
    result = agent.dep_check_node(make_state())
    assert result == {"new_deps": ["pandas"]}


def test_dep_check_node_missing_pyproject(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        agent.dep_check_node(make_state())


def test_dep_check_node_missing_lock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        agent.dep_check_node(make_state())
