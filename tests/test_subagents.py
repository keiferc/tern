import pathlib
import unittest.mock

import langchain.messages as lc_msg
import pytest

import tern.config as tern_config
import tern.subagents as subagents
import tern.tools as tern_tools


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_tern_dir(tmp_path: pathlib.Path, override: str = "") -> pathlib.Path:
    (tmp_path / "CONSTITUTION.md").write_text("# Constitution\nrule 1")
    (tmp_path / "planner.md").write_text(override)
    (tmp_path / "maker.md").write_text("")
    (tmp_path / "checker.md").write_text("")
    (tmp_path / "summarizer.md").write_text("")
    return tmp_path


def make_config() -> tern_config.Config:
    return tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20, "maker_checker_cycles": 3},
    )


def _make_mock_model(responses: list[object]) -> unittest.mock.MagicMock:
    mock_model = unittest.mock.MagicMock()
    mock_model_with_tools = unittest.mock.MagicMock()
    mock_model.bind_tools.return_value = mock_model_with_tools
    mock_model_with_tools.invoke.side_effect = responses
    return mock_model


def _make_mock_model_no_tools(responses: list[object]) -> unittest.mock.MagicMock:
    mock_model = unittest.mock.MagicMock()
    mock_model.invoke.side_effect = responses
    return mock_model


def _mock_response(
    content: str, tool_calls: list | None = None
) -> unittest.mock.MagicMock:
    resp = unittest.mock.MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


# ── _build_system_prompt ──────────────────────────────────────────────────────


def test_build_system_prompt_missing_constitution_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="CONSTITUTION.md"):
        subagents._build_system_prompt(tmp_path, "planner")


def test_build_system_prompt_constitution_only(tmp_path: pathlib.Path):
    tern_dir = _write_tern_dir(tmp_path, override="")
    result = subagents._build_system_prompt(tern_dir, "planner")
    assert result == "# Constitution\nrule 1"


def test_build_system_prompt_concatenates_non_empty_override(tmp_path: pathlib.Path):
    tern_dir = _write_tern_dir(tmp_path, override="You are a planner.")
    result = subagents._build_system_prompt(tern_dir, "planner")
    assert result == "# Constitution\nrule 1\n\nYou are a planner."


def test_build_system_prompt_whitespace_only_override_ignored(tmp_path: pathlib.Path):
    tern_dir = _write_tern_dir(tmp_path, override="   \n  ")
    result = subagents._build_system_prompt(tern_dir, "planner")
    assert result == "# Constitution\nrule 1"


# ── _extract_content ──────────────────────────────────────────────────────────


def test_extract_content_str():
    mock_resp = unittest.mock.MagicMock()
    mock_resp.content = "Here is the plan"
    assert subagents._extract_content(mock_resp) == "Here is the plan"


def test_extract_content_list():
    mock_resp = unittest.mock.MagicMock()
    mock_resp.content = [{"text": "Hello"}, " world", {"text": "!"}]
    assert subagents._extract_content(mock_resp) == "Hello world!"


# ── tools ─────────────────────────────────────────────────────────────────────


def test_read_file_reads_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.py").write_text("print('hello')")
    assert tern_tools.read_file.invoke({"path": "hello.py"}) == "print('hello')"


def test_read_file_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        tern_tools.read_file.invoke({"path": "../outside.txt"})


def test_list_files_lists_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    subdir = tmp_path / "src"
    subdir.mkdir()
    (subdir / "a.py").write_text("")
    (subdir / "b.py").write_text("")
    result = tern_tools.list_files.invoke({"path": "src"})
    assert "src/a.py" in result
    assert "src/b.py" in result


def test_list_files_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        tern_tools.list_files.invoke({"path": "../outside"})


# ── _execute_tool_calls ───────────────────────────────────────────────────────


def test_execute_tool_calls_appends_tool_result():
    tool = unittest.mock.MagicMock()
    tool.invoke.return_value = "result"
    messages: list[object] = []
    subagents._execute_tool_calls(
        {"my_tool": tool},
        [{"name": "my_tool", "args": {"x": 1}, "id": "tc1"}],
        messages,
    )
    assert len(messages) == 1
    assert isinstance(messages[0], lc_msg.ToolMessage)
    assert messages[0].content == "result"


def test_execute_tool_calls_appends_error_on_tool_exception():
    tool = unittest.mock.MagicMock()
    tool.invoke.side_effect = ValueError("disk full")
    messages: list[object] = []
    subagents._execute_tool_calls(
        {"my_tool": tool},
        [{"name": "my_tool", "args": {}, "id": "tc1"}],
        messages,
    )
    assert len(messages) == 1
    assert isinstance(messages[0], lc_msg.ToolMessage)
    assert "disk full" in messages[0].content


def test_execute_tool_calls_appends_error_for_unknown_tool():
    messages: list[object] = []
    subagents._execute_tool_calls(
        {},
        [{"name": "no_such_tool", "args": {}, "id": "tc1"}],
        messages,
    )
    assert len(messages) == 1
    assert isinstance(messages[0], lc_msg.ToolMessage)
    assert "unknown tool" in messages[0].content


# ── _react_loop ───────────────────────────────────────────────────────────────


def test_react_loop_raises_runtime_error_when_exhausted_with_pending_tool_calls():
    always_tool = _mock_response(
        content="partial",
        tool_calls=[{"name": "some_tool", "args": {}, "id": "tc1"}],
    )
    mock_model = unittest.mock.MagicMock()
    mock_model.invoke.side_effect = [always_tool, always_tool]

    with pytest.raises(RuntimeError, match="max_iterations exhausted"):
        subagents._react_loop(mock_model, {}, [unittest.mock.MagicMock()], 2, "agent")


def test_react_loop_converts_tool_exception_to_error_message():
    failing_tool = unittest.mock.MagicMock()
    failing_tool.invoke.side_effect = ValueError("disk full")
    tool_resp = _mock_response(
        content="",
        tool_calls=[{"name": "failing_tool", "args": {}, "id": "tc1"}],
    )
    final_resp = _mock_response("done")
    mock_model = unittest.mock.MagicMock()
    mock_model.invoke.side_effect = [tool_resp, final_resp]
    messages: list[object] = [unittest.mock.MagicMock()]

    subagents._react_loop(
        mock_model, {"failing_tool": failing_tool}, messages, 3, "agent"
    )

    tool_messages = [m for m in messages if isinstance(m, lc_msg.ToolMessage)]
    assert len(tool_messages) == 1
    assert "disk full" in tool_messages[0].content


def test_react_loop_appends_error_for_unknown_tool():
    tool_resp = _mock_response(
        content="",
        tool_calls=[{"name": "no_such_tool", "args": {}, "id": "tc1"}],
    )
    final_resp = _mock_response("done")
    mock_model = unittest.mock.MagicMock()
    mock_model.invoke.side_effect = [tool_resp, final_resp]
    messages: list[object] = [unittest.mock.MagicMock()]

    subagents._react_loop(mock_model, {}, messages, 3, "agent")

    tool_messages = [m for m in messages if isinstance(m, lc_msg.ToolMessage)]
    assert len(tool_messages) == 1
    assert "unknown tool" in tool_messages[0].content


# ── planner_subagent ──────────────────────────────────────────────────────────


def test_planner_subagent_returns_str_on_no_tool_calls(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("Here is the plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.planner_subagent(
            "build a classifier", make_config(), tmp_path
        )
    assert result == "Here is the plan"


def test_planner_subagent_executes_tool_call_and_continues(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    _write_tern_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spec.txt").write_text("use pandas")

    tool_resp = _mock_response(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "spec.txt"}, "id": "tc1"}],
    )
    final_resp = _mock_response("Final plan")
    mock_model = _make_mock_model([tool_resp, final_resp])

    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.planner_subagent(
            "build a classifier", make_config(), tmp_path
        )

    assert result == "Final plan"
    assert mock_model.bind_tools.return_value.invoke.call_count == 2


def test_planner_subagent_includes_prior_plan_in_messages(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("revised plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier", make_config(), tmp_path, prior_plan="old plan"
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == "old plan"


@pytest.mark.parametrize("feedback", [None, []], ids=["none", "empty"])
def test_planner_subagent_omits_feedback_when_absent(
    feedback: list | None, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier", make_config(), tmp_path, feedback=feedback
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    human_msgs = [m for m in messages if isinstance(m, lc_msg.HumanMessage)]
    assert not any("Prior Feedback" in m.content for m in human_msgs)


def test_planner_subagent_includes_feedback_section(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier",
            make_config(),
            tmp_path,
            feedback=["fix the imports", "add type hints"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    human_msgs = [m for m in messages if isinstance(m, lc_msg.HumanMessage)]
    feedback_msg = next(m for m in human_msgs if "Prior Feedback" in m.content)
    assert "fix the imports" in feedback_msg.content
    assert "add type hints" in feedback_msg.content


def test_planner_subagent_message_order_with_prior_plan_and_feedback(
    tmp_path: pathlib.Path,
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier",
            make_config(),
            tmp_path,
            prior_plan="old plan",
            feedback=["fix the imports"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    assert isinstance(messages[0], lc_msg.SystemMessage)
    assert isinstance(messages[1], lc_msg.HumanMessage)
    assert isinstance(messages[2], lc_msg.AIMessage)
    assert isinstance(messages[3], lc_msg.HumanMessage)
    assert "Prior Feedback" in messages[3].content


@pytest.mark.parametrize("issues", [None, []], ids=["none", "empty"])
def test_planner_subagent_omits_issues_when_absent(
    issues: list | None, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier", make_config(), tmp_path, issues=issues
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    assert not any("Checker Issues" in m.content for m in ai_msgs)


def test_planner_subagent_includes_checker_issues_section(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier",
            make_config(),
            tmp_path,
            issues=["unused import on line 3", "missing type hint on foo"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    issues_msg = next(m for m in ai_msgs if "Checker Issues" in m.content)
    assert "unused import on line 3" in issues_msg.content
    assert "missing type hint on foo" in issues_msg.content


def test_planner_subagent_message_order_with_all_context(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("plan")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.planner_subagent(
            "build a classifier",
            make_config(),
            tmp_path,
            prior_plan="old plan",
            issues=["unused import"],
            feedback=["fix the imports"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    assert isinstance(messages[0], lc_msg.SystemMessage)
    assert isinstance(messages[1], lc_msg.HumanMessage)  # objective
    assert isinstance(messages[2], lc_msg.AIMessage)  # prior_plan
    assert isinstance(messages[3], lc_msg.AIMessage)  # issues
    assert "Checker Issues" in messages[3].content
    assert isinstance(messages[4], lc_msg.HumanMessage)  # feedback
    assert "Prior Feedback" in messages[4].content


@pytest.mark.parametrize(
    "max_iterations",
    [{"default": 0}, {"default": 20, "planner": 0}],
    ids=["default", "per_agent"],
)
def test_planner_subagent_raises_if_max_iterations_zero(
    max_iterations: dict, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations=max_iterations,
    )
    mock_model = _make_mock_model([])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="max_iterations"):
            subagents.planner_subagent("build a classifier", config, tmp_path)


# ── maker_subagent ────────────────────────────────────────────────────────────


def test_maker_subagent_returns_written_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    _write_tern_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    tool_resp = _mock_response(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": "output.py", "content": "x = 1"},
                "id": "tc1",
            }
        ],
    )
    final_resp = _mock_response("done")
    mock_model = _make_mock_model([tool_resp, final_resp])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.maker_subagent(
            "build a classifier", "step 1", make_config(), tmp_path
        )
    assert result == [str((tmp_path / "output.py").resolve())]
    assert (tmp_path / "output.py").read_text() == "x = 1"


def test_maker_subagent_includes_issues_section(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("done")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.maker_subagent(
            "build a classifier",
            "step 1",
            make_config(),
            tmp_path,
            issues=["unused import on line 3"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    issues_msg = next(m for m in ai_msgs if "Checker Issues" in m.content)
    assert "unused import on line 3" in issues_msg.content


@pytest.mark.parametrize("issues", [None, []], ids=["none", "empty"])
def test_maker_subagent_omits_issues_section_when_absent(
    issues: list | None, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("done")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.maker_subagent(
            "build a classifier", "step 1", make_config(), tmp_path, issues=issues
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    assert not any("Checker Issues" in m.content for m in ai_msgs)


def test_maker_subagent_includes_feedback_section(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("done")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.maker_subagent(
            "build a classifier",
            "step 1",
            make_config(),
            tmp_path,
            feedback=["add type hints", "avoid pandas"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    human_msgs = [m for m in messages if isinstance(m, lc_msg.HumanMessage)]
    feedback_msg = next(m for m in human_msgs if "Prior Feedback" in m.content)
    assert "add type hints" in feedback_msg.content
    assert "avoid pandas" in feedback_msg.content


@pytest.mark.parametrize("feedback", [None, []], ids=["none", "empty"])
def test_maker_subagent_omits_feedback_section_when_absent(
    feedback: list | None, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("done")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.maker_subagent(
            "build a classifier", "step 1", make_config(), tmp_path, feedback=feedback
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    human_msgs = [m for m in messages if isinstance(m, lc_msg.HumanMessage)]
    assert not any("Prior Feedback" in m.content for m in human_msgs)


def test_maker_subagent_message_order_with_all_context(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("done")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.maker_subagent(
            "build a classifier",
            "step 1: build model",
            make_config(),
            tmp_path,
            issues=["unused import"],
            feedback=["add type hints"],
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    assert isinstance(messages[0], lc_msg.SystemMessage)
    assert isinstance(messages[1], lc_msg.HumanMessage)  # objective
    assert isinstance(messages[2], lc_msg.AIMessage)  # plan
    assert isinstance(messages[3], lc_msg.AIMessage)  # issues
    assert "Checker Issues" in messages[3].content
    assert isinstance(messages[4], lc_msg.HumanMessage)  # feedback
    assert "Prior Feedback" in messages[4].content


# ── checker_subagent ──────────────────────────────────────────────────────────


def test_checker_subagent_empty_response_returns_empty_list(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.checker_subagent("", "", make_config(), tmp_path)
    assert result == []


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            "issue one\nissue two\n\nissue three",
            ["issue one", "issue two", "issue three"],
        ),
        ("\n\n  \nissue one\n  \n", ["issue one"]),
    ],
    ids=["multiline", "blank_lines_filtered"],
)
def test_checker_subagent_output_parsed(
    content: str, expected: list, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response(content)])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.checker_subagent("", "", make_config(), tmp_path)
    assert result == expected


def test_checker_subagent_executes_tool_call_and_continues(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    _write_tern_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "module.py").write_text("x = 1")

    tool_resp = _mock_response(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "module.py"}, "id": "tc1"}],
    )
    final_resp = _mock_response("issue one")
    mock_model = _make_mock_model([tool_resp, final_resp])

    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.checker_subagent("", "", make_config(), tmp_path)

    assert result == ["issue one"]
    assert mock_model.bind_tools.return_value.invoke.call_count == 2


@pytest.mark.parametrize(
    "max_iterations",
    [{"default": 0}, {"default": 20, "checker": 0}],
    ids=["default", "per_agent"],
)
def test_checker_subagent_raises_if_max_iterations_zero(
    max_iterations: dict, tmp_path: pathlib.Path
):
    _write_tern_dir(tmp_path)
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations=max_iterations,
    )
    mock_model = _make_mock_model([])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="max_iterations"):
            subagents.checker_subagent("", "", config, tmp_path)


def test_checker_subagent_human_message_contains_qa_output(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.checker_subagent(
            "ruff: all good", "=== foo.py ===\nx=1", make_config(), tmp_path
        )
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    human_content = messages[1].content
    assert isinstance(messages[1], lc_msg.HumanMessage)
    assert "ruff: all good" in human_content
    assert "no preamble" in human_content


def test_checker_subagent_ai_message_contains_file_contents(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.checker_subagent("", "=== foo.py ===\nx=1", make_config(), tmp_path)
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    assert len(ai_msgs) == 1
    assert "=== foo.py ===" in ai_msgs[0].content


def test_checker_subagent_omits_ai_message_when_no_files(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.checker_subagent("", "", make_config(), tmp_path)
    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    ai_msgs = [m for m in messages if isinstance(m, lc_msg.AIMessage)]
    assert len(ai_msgs) == 0


# ── summarizer_subagent ───────────────────────────────────────────────────────


def test_summarizer_subagent_empty_state_returns_empty_without_calling_model(
    tmp_path: pathlib.Path,
):
    with unittest.mock.patch("tern.models.get_model") as mock_get:
        result = subagents.summarizer_subagent(
            {"objective": None, "plan": None, "written_files": [], "messages": []},
            make_config(),
            tmp_path,
        )
    assert result == ""
    mock_get.assert_not_called()


def test_summarizer_subagent_includes_objective_in_prompt(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(
            {"objective": "build a classifier", "messages": []}, make_config(), tmp_path
        )
    human_content = mock_model.invoke.call_args[0][0][1].content
    assert "build a classifier" in human_content


def test_summarizer_subagent_includes_plan_in_prompt(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(
            {"plan": "step 1: train model", "messages": []}, make_config(), tmp_path
        )
    messages = mock_model.invoke.call_args[0][0]
    assert isinstance(messages[2], lc_msg.AIMessage)
    assert "step 1: train model" in messages[2].content


def test_summarizer_subagent_includes_written_files_in_prompt(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(
            {"written_files": ["src/model.py"], "messages": []}, make_config(), tmp_path
        )
    human_content = mock_model.invoke.call_args[0][0][1].content
    assert "src/model.py" in human_content


def test_summarizer_subagent_filters_to_human_messages_only(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    state = {
        "objective": "build a model",
        "messages": [
            lc_msg.HumanMessage(content="user turn"),
            lc_msg.AIMessage(content="<<AI_ONLY_SENTINEL>>"),
        ],
    }
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(state, make_config(), tmp_path)
    human_content = mock_model.invoke.call_args[0][0][1].content
    assert "user turn" in human_content
    assert "<<AI_ONLY_SENTINEL>>" not in human_content


def test_summarizer_subagent_skips_plan_section_when_plan_absent(
    tmp_path: pathlib.Path,
):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(
            {"objective": "build a model", "messages": []},
            make_config(),
            tmp_path,
        )
    messages = mock_model.invoke.call_args[0][0]
    assert len(messages) == 2  # no AIMessage when plan absent


def test_summarizer_subagent_skips_empty_human_message(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    state = {
        "objective": "build a model",
        "messages": [lc_msg.HumanMessage(content="")],
    }
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(state, make_config(), tmp_path)
    human_content = mock_model.invoke.call_args[0][0][1].content
    assert "## User Message" not in human_content


# ── read_file sensitive-file blocklist ────────────────────────────────────────


@pytest.mark.parametrize(
    "filename",
    [
        ".env",
        ".env.local",
        "server.key",
        "cert.pem",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "aws_credentials",
        "my_secret",
        "access_token",
    ],
)
def test_read_file_raises_for_sensitive_filename(
    filename: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / filename).write_text("secret")
    with pytest.raises(ValueError, match="sensitive-file pattern"):
        tern_tools.read_file.invoke({"path": filename})


def test_read_file_allows_env_example(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("API_KEY=changeme")
    result = tern_tools.read_file.invoke({"path": ".env.example"})
    assert result == "API_KEY=changeme"


def test_read_file_allows_tokenizer_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tokenizer.py").write_text("# tokenizer")
    result = tern_tools.read_file.invoke({"path": "tokenizer.py"})
    assert result == "# tokenizer"


# ── web_fetch ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/file"])
def test_web_fetch_rejects_non_http_scheme(url: str):
    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        result = tern_tools.web_fetch.invoke({"url": url})
    assert result.startswith("Error:")
    mock_urlopen.assert_not_called()


def test_web_fetch_passes_timeout_to_urlopen():
    mock_resp = unittest.mock.MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"hello"
    with unittest.mock.patch(
        "urllib.request.urlopen", return_value=mock_resp
    ) as mock_open:
        tern_tools.web_fetch.invoke({"url": "https://example.com"})
    mock_open.assert_called_once_with("https://example.com", timeout=30)


def test_web_fetch_propagates_urlopen_failure():
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=OSError("connection refused")
    ):
        with pytest.raises(OSError, match="connection refused"):
            tern_tools.web_fetch.invoke({"url": "https://example.com"})


def test_web_fetch_truncates_long_response():
    content = b"x" * 25000
    mock_resp = unittest.mock.MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = content
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = tern_tools.web_fetch.invoke({"url": "https://example.com"})
    assert result[:20000] == "x" * 20000
    assert result.endswith("\n[... truncated]")


def test_web_fetch_does_not_truncate_short_response():
    content = b"hello world"
    mock_resp = unittest.mock.MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = content
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = tern_tools.web_fetch.invoke({"url": "https://example.com"})
    assert result == "hello world"
