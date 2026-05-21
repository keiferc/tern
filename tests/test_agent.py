import operator
import pathlib
import typing as T
import unittest.mock

import langchain.messages as lc_msg
import langgraph.graph as lg_graph

import tern.agent as agent
import tern.config as tern_config
import tern.subagents as tern_subagents


def make_config() -> tern_config.Config:
    return tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20},
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
            "messages": [],
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


def test_route_user_no_objective_with_handoff():
    state = make_state(objective=None, need_handoff=True)
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


def test_route_user_new_deps_no_decision():
    state = make_state(
        objective="build a model",
        plan_approved=True,
        new_deps=["pandas"],
        deps_approved=None,
    )
    assert agent.route_from_user(state) == "user"


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
    state = make_state(issues=["unused import on line 3"])
    assert agent.route_from_checker(state) == "maker"


def test_route_checker_no_issues():
    state = make_state(issues=[])
    assert agent.route_from_checker(state) == "user"


# ── nodes ─────────────────────────────────────────────────────────────────


def test_user_node_returns_empty_dict():
    state = make_state()
    assert agent.user_node(state) == {}


def test_planner_node_returns_plan_fields(tmp_path: pathlib.Path):
    state = make_state(objective="build a model")
    with unittest.mock.patch(
        "tern.subagents.planner_subagent", return_value="step 1: do thing"
    ):
        result = agent.planner_node(state, make_config(), tmp_path)
    assert "plan" in result
    assert result["plan"] == "step 1: do thing"
    assert "plan_approved" in result
    assert result["plan_approved"] is None


def test_maker_node_returns_written_files(tmp_path: pathlib.Path):
    state = make_state(plan="step 1: do thing")
    with unittest.mock.patch.object(
        tern_subagents, "maker_subagent", return_value=["foo.py"]
    ):
        result = agent.maker_node(state, make_config(), tmp_path)
    assert result == {"written_files": ["foo.py"]}


def test_dep_check_graph_node_returns_new_deps(tmp_path: pathlib.Path):
    state = make_state()
    result = agent.dep_check_graph_node(state, make_config(), tmp_path)
    assert "new_deps" in result
    assert isinstance(result["new_deps"], list)


def test_qa_runner_graph_node_returns_qa_output(tmp_path: pathlib.Path):
    state = make_state()
    result = agent.qa_runner_graph_node(state, make_config(), tmp_path)
    assert "qa_output" in result
    assert isinstance(result["qa_output"], str)


def test_checker_graph_node_returns_issues(tmp_path: pathlib.Path):
    state = make_state(qa_output="ruff: 0 errors", written_files=[])
    result = agent.checker_graph_node(state, make_config(), tmp_path)
    assert "issues" in result
    assert isinstance(result["issues"], list)


def test_checker_graph_node_formats_written_files(tmp_path: pathlib.Path):
    (tmp_path / "foo.py").write_text("x = 1")
    state = make_state(qa_output="", written_files=[str(tmp_path / "foo.py")])
    with unittest.mock.patch.object(
        tern_subagents, "checker_subagent", return_value=[]
    ) as mock_checker:
        agent.checker_graph_node(state, make_config(), tmp_path)
    file_contents_arg = mock_checker.call_args[0][1]
    assert "foo.py" in file_contents_arg
    assert "x = 1" in file_contents_arg


def test_checker_graph_node_skips_missing_files(tmp_path: pathlib.Path):
    state = make_state(qa_output="", written_files=[str(tmp_path / "ghost.py")])
    result = agent.checker_graph_node(state, make_config(), tmp_path)
    assert result == {"issues": []}


def test_summarizer_graph_node_returns_empty_dict(tmp_path: pathlib.Path):
    state = make_state()
    assert agent.summarizer_graph_node(state, make_config(), tmp_path) == {}


def test_summarizer_graph_node_writes_handoff_doc(tmp_path: pathlib.Path):
    state = make_state()
    with unittest.mock.patch.object(
        tern_subagents, "summarizer_subagent", return_value="# Handoff\n\nDone."
    ):
        with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            agent.summarizer_graph_node(state, make_config(), tmp_path)
    assert (tmp_path / "HANDOFF.md").read_text() == "# Handoff\n\nDone."


def test_summarizer_graph_node_skips_write_when_empty(tmp_path: pathlib.Path):
    state = make_state()
    with unittest.mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        agent.summarizer_graph_node(state, make_config(), tmp_path)
    assert not (tmp_path / "HANDOFF.md").exists()


# ── messages accumulation ─────────────────────────────────────────────────


def test_messages_accumulate_via_operator_add():
    msgs_a = [lc_msg.HumanMessage(content="hello")]
    msgs_b = [lc_msg.AIMessage(content="world")]
    result = operator.add(msgs_a, msgs_b)
    assert len(result) == 2
    assert result[0].content == "hello"
    assert result[1].content == "world"


def test_messages_field_uses_add_reducer():
    # Annotated[list[AnyMessage], operator.add] means updates append, not replace.
    # Verify the annotation is in place on the AgentState class.
    annotated_args = T.get_args(agent.AgentState.__annotations__["messages"])
    assert operator.add in annotated_args
