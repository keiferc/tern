import pathlib
import typing as T

import langchain.messages as lc_msg

import tern.config as tern_config
import tern.models as tern_models
import tern.tools as tern_tools


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
    tools = [tern_tools.web_fetch, tern_tools.read_file, tern_tools.list_files]
    model = tern_models.get_model(config, "planner").bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "planner")),
        lc_msg.HumanMessage(content=objective),
    ]
    if prior_plan:
        messages.append(lc_msg.AIMessage(content=prior_plan))
    if issues:
        messages.append(
            lc_msg.AIMessage(
                content="## Checker Issues\n" + "\n".join(f"- {i}" for i in issues)
            )
        )
    if feedback:
        messages.append(
            lc_msg.HumanMessage(content="## Prior Feedback\n" + "\n".join(feedback))
        )
    response = _react_loop(
        model, tool_map, messages, _max_iter(config, "planner"), "planner_subagent"
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
    tools = [
        tern_tools.read_file,
        tern_tools.write_file,
        tern_tools.list_files,
        tern_tools.web_fetch,
    ]
    model = tern_models.get_model(config, "maker").bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "maker")),
        lc_msg.HumanMessage(content=objective),
        lc_msg.AIMessage(content=plan),
    ]
    if issues:
        messages.append(
            lc_msg.AIMessage(
                content="## Checker Issues\n" + "\n".join(f"- {i}" for i in issues)
            )
        )
    if feedback:
        messages.append(
            lc_msg.HumanMessage(content="## Prior Feedback\n" + "\n".join(feedback))
        )
    _react_loop(model, tool_map, messages, _max_iter(config, "maker"), "maker_subagent")
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
) -> list[str]:
    tools = [tern_tools.web_fetch, tern_tools.read_file, tern_tools.list_files]
    model = tern_models.get_model(config, "checker").bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    if file_contents:
        preamble = (
            "The following files were written by the maker. These are your primary review target.\n"
            "Use tools only to read additional project files or verify documentation — "
            "do not re-read files already provided here.\n"
        )
    else:
        preamble = "Review the QA output below for issues.\n"
    task_instruction = (
        f"{preamble}\n"
        f"## QA Tool Output\n"
        f"{qa_output}\n"
        "\n"
        "Report each issue on its own line. "
        "Output only issues — no preamble, no summary, no explanation."
    )

    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "checker")),
        lc_msg.HumanMessage(content=task_instruction),
    ]
    if file_contents:
        messages.append(lc_msg.AIMessage(content=file_contents))
    response = _react_loop(
        model, tool_map, messages, _max_iter(config, "checker"), "checker_subagent"
    )
    content = _extract_content(response)
    return [line for line in (ln.strip() for ln in content.splitlines()) if line]


def summarizer_subagent(
    state: dict, config: tern_config.Config, tern_dir: pathlib.Path
) -> str:
    human_parts: list[str] = []

    if state.get("objective"):
        human_parts.append(f"## Objective\n{state['objective']}")
    if state.get("written_files"):
        human_parts.append("## Written Files\n" + "\n".join(state["written_files"]))
    for msg in state.get("messages", []):
        if isinstance(msg, lc_msg.HumanMessage):
            text = _extract_content(msg)
            if text:
                human_parts.append(f"## User Message\n{text}")

    plan = state.get("plan")

    if not human_parts and not plan:
        return ""

    human_content = "Summarize the following session for handoff."
    if human_parts:
        human_content += "\n\n" + "\n\n".join(human_parts)

    model = tern_models.get_model(config, "summarizer")
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "summarizer")),
        lc_msg.HumanMessage(content=human_content),
    ]
    if plan:
        messages.append(lc_msg.AIMessage(content=f"## Plan\n{plan}"))
    response = model.invoke(messages)  # ty: ignore[invalid-argument-type]
    return _extract_content(response)


# ========================================================================= #
#                                                                           #
#                               Helpers                                     #
#                                                                           #
# ========================================================================= #


def _max_iter(config: tern_config.Config, agent: str) -> int:
    _val = config.max_iterations.get(agent)
    return config.max_iterations["default"] if _val is None else _val


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


def _extract_content(response: object) -> str:
    # AIMessage.content is str (OpenAI) or list of blocks (Anthropic); normalise to str.
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "") for block in content
    )


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
    response: object = None
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
