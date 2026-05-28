import pathlib
import typing as T

import langchain.messages as lc_msg
import langchain_core.tools as lc_tools

import tern.config as tern_config
import tern.models as tern_models
import tern.tools as tern_tools

_NO_ISSUES_PHRASES: frozenset[str] = frozenset(
    {
        "clean",
        "lgtm",
        "no issue",
        "no issue found",
        "no issues",
        "no issues detected",
        "no issues found",
        "none",
        "pass",
        "passed",
    }
)


# ========================================================================= #
#                                                                           #
#                               Subagents                                   #
#                                                                           #
# ========================================================================= #


def planner_subagent(
    objective: str,
    config: tern_config.Config,
    tern_dir: pathlib.Path,
    *,
    prior_plan: str | None = None,
    issues: list[str] | None = None,
    feedback: list[str] | None = None,
) -> str:
    tools: T.Sequence[lc_tools.BaseTool] = [
        tern_tools.web_fetch,
        tern_tools.read_file,
        tern_tools.list_files,
    ]
    model = tern_models.get_model(config, "planner").bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "planner")),
        lc_msg.HumanMessage(content=objective),
    ]
    if prior_plan:
        messages.append(lc_msg.AIMessage(content=prior_plan))
    _append_context(messages, issues, feedback)
    response = _react_loop(
        model,
        tool_map,
        messages,
        config.max_iterations["planner"],
        "planner_subagent",
    )
    return _extract_content(response)


def maker_subagent(
    objective: str,
    plan: str,
    config: tern_config.Config,
    tern_dir: pathlib.Path,
    *,
    issues: list[str] | None = None,
    feedback: list[str] | None = None,
) -> list[str]:
    tools: T.Sequence[lc_tools.BaseTool] = [
        tern_tools.read_file,
        tern_tools.write_file,
        tern_tools.list_files,
        tern_tools.web_fetch,
    ]
    model = tern_models.get_model(config, "maker").bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "maker")),
        lc_msg.HumanMessage(
            content=f"Objective: {objective}\n\n## Approved Plan\n{plan}"
        ),
    ]
    _append_context(messages, issues, feedback)

    response = _react_loop(
        model,
        tool_map,
        messages,
        config.max_iterations["maker"],
        "maker_subagent",
    )
    _ = _extract_content(response)

    tc_names = {
        tc["id"]: tc["name"]
        for msg in messages
        for tc in getattr(msg, "tool_calls", [])
    }
    return [
        msg.content
        for msg in messages
        if isinstance(msg, lc_msg.ToolMessage)
        and tc_names.get(msg.tool_call_id) == "write_file"
        and isinstance(msg.content, str)
        and not msg.content.startswith("Error:")
    ]


def checker_subagent(
    qa_output: str,
    file_contents: str,
    config: tern_config.Config,
    tern_dir: pathlib.Path,
    *,
    plan: str | None = None,
    feedback: list[str] | None = None,
) -> list[str]:
    tools: T.Sequence[lc_tools.BaseTool] = [
        tern_tools.web_fetch,
        tern_tools.read_file,
        tern_tools.list_files,
    ]
    model = tern_models.get_model(config, "checker").bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    if file_contents:
        preamble = (
            "no preamble. "
            "The files written by the maker are provided below under ## Written Files. "
            "These are your primary review target.\n"
            "Use tools only to read additional project files or verify documentation — "
            "do not re-read files already provided here.\n"
        )
    else:
        preamble = "no preamble. Review the QA output below for issues.\n"

    task_instruction = f"{preamble}\n## QA Tool Output\n{qa_output}\n"
    if plan:
        task_instruction += f"\n## Approved Plan\n{plan}\n"
    if feedback:
        task_instruction += "\n## Session Feedback\n" + "\n".join(feedback) + "\n"
    if file_contents:
        task_instruction += f"\n## Written Files\n{file_contents}\n"
    task_instruction += (
        "\nReport each issue on its own line. "
        "If there are no issues, output nothing — an empty response."
    )

    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "checker")),
        lc_msg.HumanMessage(content=task_instruction),
    ]
    response = _react_loop(
        model,
        tool_map,
        messages,
        config.max_iterations["checker"],
        "checker_subagent",
    )
    content = _extract_content(response)
    return [
        line
        for line in (ln.strip() for ln in content.splitlines())
        if line and not _is_no_issue_line(line)
    ]


def summarizer_subagent(
    state: dict,
    file_contents: str,
    config: tern_config.Config,
    tern_dir: pathlib.Path,
) -> str:
    human_parts: list[str] = []

    objectives = state.get("session_objectives") or []
    if objectives:
        numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(objectives))
        human_parts.append(f"## Objectives\n{numbered}")
    if file_contents:
        human_parts.append(f"## Files Written\n{file_contents}")
    for milestone in state.get("milestones", []):
        if milestone:
            human_parts.append(f"## Completed Plan\n{milestone}")

    plan = state.get("plan")
    if plan:
        human_parts.append(f"## Last Plan\n{plan}")

    if not human_parts:
        return ""

    human_content = "Summarize the following session for handoff.\n\n" + "\n\n".join(
        human_parts
    )

    model = tern_models.get_model(config, "summarizer")
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "summarizer")),
        lc_msg.HumanMessage(content=human_content),
    ]

    response = model.invoke(messages)  # ty: ignore[invalid-argument-type]
    return _extract_content(response)


# ========================================================================= #
#                                                                           #
#                               Helpers                                     #
#                                                                           #
# ========================================================================= #


def _append_context(
    messages: list[object],
    issues: list[str] | None,
    feedback: list[str] | None,
) -> None:
    parts: list[str] = []
    if issues:
        parts.append("## Checker Issues\n" + "\n".join(f"- {i}" for i in issues))
    if feedback:
        parts.append("## Prior Feedback\n" + "\n".join(feedback))
    if parts:
        messages.append(lc_msg.HumanMessage(content="\n\n".join(parts)))


def _build_system_prompt(tern_dir: pathlib.Path, agent: str) -> str:
    path = tern_dir / "CONSTITUTION.md"
    try:
        constitution = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"CONSTITUTION.md not found in tern directory: {path}")
    override = tern_config.load_agent_prompt(tern_dir, agent)
    if override:
        return f"{constitution}\n\n{override}"
    return constitution


def _is_no_issue_line(line: str) -> bool:
    return line.lower().rstrip("!:. ").strip() in _NO_ISSUES_PHRASES


def _extract_content(response: object) -> str:
    # AIMessage.content is str (OpenAI) or list of blocks (Anthropic); normalise to str.
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return ""


def _execute_tool_calls(
    tool_map: dict,
    tool_calls: list,
    messages: list[object],
) -> None:
    for tool_call in tool_calls:
        tool = tool_map.get(tool_call["name"])
        if tool is None:
            result = f"Error: unknown tool {tool_call['name']!r}"
        else:
            try:
                result = str(tool.invoke(tool_call["args"]))
            except Exception as exc:
                result = f"Error: {exc}"
        messages.append(
            lc_msg.ToolMessage(content=result, tool_call_id=tool_call["id"])
        )


def _react_loop(
    model: T.Any,
    tool_map: dict,
    messages: list[object],
    max_iter: int,
    agent_name: str,
) -> object:
    response: object | None = None
    for _ in range(max_iter):
        response = model.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            break
        _execute_tool_calls(tool_map, tool_calls, messages)
    if response is None:
        raise ValueError(
            f"{agent_name} produced no response: max_iterations is {max_iter}"
        )
    if getattr(response, "tool_calls", []):
        raise RuntimeError(
            f"{agent_name}: max_iterations exhausted with pending tool calls"
        )
    return response
