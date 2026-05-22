import pathlib
import unittest.mock

import langchain.messages as lc_msg
import pytest

import tern.config as tern_config
import tern.subagents as subagents


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_tern_dir(tmp_path: pathlib.Path, override: str = "") -> pathlib.Path:
    (tmp_path / "CONSTITUTION.md").write_text("# Constitution\nrule 1")
    (tmp_path / "planner.md").write_text(override)
    (tmp_path / "maker.md").write_text("")
    (tmp_path / "checker.md").write_text("")
    (tmp_path / "summarizer.md").write_text("")
    return tmp_path


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


# ── fixtures ──────────────────────────────────────────────────────────────────


def make_config() -> tern_config.Config:
    return tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20},
    )


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
    assert subagents.read_file.invoke({"path": "hello.py"}) == "print('hello')"


def test_read_file_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        subagents.read_file.invoke({"path": "../outside.txt"})


def test_list_files_lists_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    subdir = tmp_path / "src"
    subdir.mkdir()
    (subdir / "a.py").write_text("")
    (subdir / "b.py").write_text("")
    result = subagents.list_files.invoke({"path": "src"})
    assert "src/a.py" in result
    assert "src/b.py" in result


def test_list_files_raises_outside_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        subagents.list_files.invoke({"path": "../outside"})


# ── planner_subagent ──────────────────────────────────────────────────────────


def _make_mock_model(responses: list[object]) -> unittest.mock.MagicMock:
    mock_model = unittest.mock.MagicMock()
    mock_model_with_tools = unittest.mock.MagicMock()
    mock_model.bind_tools.return_value = mock_model_with_tools
    mock_model_with_tools.invoke.side_effect = responses
    return mock_model


def _mock_response(
    content: str, tool_calls: list | None = None
) -> unittest.mock.MagicMock:
    resp = unittest.mock.MagicMock()
    resp.content = content
    resp.tool_calls = tool_calls or []
    return resp


def test_planner_subagent_calls_get_model_with_planner(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("the plan")])
    with unittest.mock.patch(
        "tern.models.get_model", return_value=mock_model
    ) as mock_get:
        subagents.planner_subagent("build a classifier", make_config(), tmp_path)
    mock_get.assert_called_once_with(make_config(), "planner")


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


def test_planner_subagent_raises_on_max_iterations_exhausted_with_tool_calls(
    tmp_path: pathlib.Path,
):
    _write_tern_dir(tmp_path)
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20, "planner": 2},
    )
    always_tool = _mock_response(
        content="partial",
        tool_calls=[{"name": "read_file", "args": {"path": "x.txt"}, "id": "tc1"}],
    )
    mock_model = _make_mock_model([always_tool, always_tool])

    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        with pytest.raises(RuntimeError, match="max_iterations exhausted"):
            subagents.planner_subagent("build a classifier", config, tmp_path)

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


def test_planner_subagent_raises_if_max_iterations_zero(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 0},
    )
    mock_model = _make_mock_model([])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="max_iterations"):
            subagents.planner_subagent("build a classifier", config, tmp_path)


def test_maker_subagent_returns_list(tmp_path: pathlib.Path):
    assert isinstance(subagents.maker_subagent("step 1", make_config(), tmp_path), list)


# ── checker_subagent ──────────────────────────────────────────────────────────


def test_checker_subagent_calls_get_model_with_checker(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])
    with unittest.mock.patch(
        "tern.models.get_model", return_value=mock_model
    ) as mock_get:
        subagents.checker_subagent("", "", make_config(), tmp_path)
    mock_get.assert_called_once_with(make_config(), "checker")


def test_checker_subagent_empty_response_returns_empty_list(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.checker_subagent("", "", make_config(), tmp_path)
    assert result == []


def test_checker_subagent_multiline_response_parsed(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model(
        [_mock_response("issue one\nissue two\n\nissue three")]
    )
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.checker_subagent(
            "ruff ok", "file content", make_config(), tmp_path
        )
    assert result == ["issue one", "issue two", "issue three"]


def test_checker_subagent_blank_lines_filtered(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("\n\n  \nissue one\n  \n")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.checker_subagent("", "", make_config(), tmp_path)
    assert result == ["issue one"]


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


def test_checker_subagent_raises_on_max_iterations_exhausted_with_tool_calls(
    tmp_path: pathlib.Path,
):
    _write_tern_dir(tmp_path)
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20, "checker": 2},
    )
    always_tool = _mock_response(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "x.txt"}, "id": "tc1"}],
    )
    mock_model = _make_mock_model([always_tool, always_tool])

    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        with pytest.raises(RuntimeError, match="max_iterations exhausted"):
            subagents.checker_subagent("", "", config, tmp_path)

    assert mock_model.bind_tools.return_value.invoke.call_count == 2


def test_checker_subagent_raises_if_max_iterations_zero(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    config = tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 0},
    )
    mock_model = _make_mock_model([])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="max_iterations"):
            subagents.checker_subagent("", "", config, tmp_path)


def test_checker_subagent_human_message_contains_qa_and_files(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model([_mock_response("")])

    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.checker_subagent(
            "ruff: all good", "=== foo.py ===\nx=1", make_config(), tmp_path
        )

    messages = mock_model.bind_tools.return_value.invoke.call_args[0][0]
    human_content = messages[1].content
    assert "ruff: all good" in human_content
    assert "=== foo.py ===" in human_content
    assert "no preamble" in human_content


def _make_mock_model_no_tools(responses: list[object]) -> unittest.mock.MagicMock:
    mock_model = unittest.mock.MagicMock()
    mock_model.invoke.side_effect = responses
    return mock_model


def test_summarizer_subagent_calls_get_model_with_summarizer(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch(
        "tern.models.get_model", return_value=mock_model
    ) as mock_get:
        subagents.summarizer_subagent(
            {"objective": "build a model", "messages": []}, make_config(), tmp_path
        )
    mock_get.assert_called_once_with(make_config(), "summarizer")


def test_summarizer_subagent_returns_str(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        result = subagents.summarizer_subagent(
            {"objective": "build a model", "messages": []}, make_config(), tmp_path
        )
    assert isinstance(result, str)


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


def test_summarizer_subagent_single_invocation(tmp_path: pathlib.Path):
    _write_tern_dir(tmp_path)
    mock_model = _make_mock_model_no_tools([_mock_response("# Handoff")])
    with unittest.mock.patch("tern.models.get_model", return_value=mock_model):
        subagents.summarizer_subagent(
            {"objective": "build a model", "messages": []}, make_config(), tmp_path
        )
    assert mock_model.invoke.call_count == 1


def test_dep_check_node_returns_list(tmp_path: pathlib.Path):
    assert isinstance(subagents.dep_check_node(make_config(), tmp_path), list)


def test_qa_runner_node_returns_str(tmp_path: pathlib.Path):
    assert isinstance(subagents.qa_runner_node(make_config(), tmp_path), str)


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


# ── web_fetch ─────────────────────────────────────────────────────────────────


def test_web_fetch_returns_error_string_on_urlopen_failure():
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=OSError("connection refused")
    ):
        result = subagents.web_fetch.invoke({"url": "https://example.com"})
    assert result.startswith("Error fetching")
    assert "connection refused" in result


# ── summarizer_subagent (coverage gaps) ──────────────────────────────────────


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
    human_content = mock_model.invoke.call_args[0][0][1].content
    assert "## Plan" not in human_content


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
