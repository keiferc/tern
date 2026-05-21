import pathlib
import urllib.request

import langchain.messages as lc_msg
import langchain_core.tools as lc_tools

import tern.config as tern_config
import tern.models as tern_models

# ========================================================================= #
#                                                                           #
#                               Helpers                                     #
#                                                                           #
# ========================================================================= #


def _build_system_prompt(tern_dir: pathlib.Path, agent: str) -> str:
    constitution = (tern_dir / "CONSTITUTION.md").read_text()
    override = tern_config.load_agent_prompt(tern_dir, agent)
    if override:
        return f"{constitution}\n\n{override}"
    return constitution


def _extract_content(response: object) -> str:
    """
    AIMessage.content is str (OpenAI) or list of blocks (Anthropic); normalise to str.

    """
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "") for block in content
    )


def _safe_resolve(path_str: str) -> pathlib.Path:
    cwd = pathlib.Path.cwd().resolve()
    resolved = (cwd / path_str).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ValueError(f"path outside working directory: {path_str!r}")
    return resolved


# ========================================================================= #
#                                                                           #
#                               Tools                                       #
#                                                                           #
# ========================================================================= #


@lc_tools.tool
def web_fetch(url: str) -> str:
    """Fetch the text content of a URL."""
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error fetching {url!r}: {exc}"


@lc_tools.tool
def read_file(path: str) -> str:
    """Read a file within the project working directory and return its content."""
    return _safe_resolve(path).read_text()


@lc_tools.tool
def list_files(path: str) -> str:
    """List files in a directory within the project working directory."""
    cwd = pathlib.Path.cwd().resolve()
    return "\n".join(
        str(p.relative_to(cwd)) for p in sorted(_safe_resolve(path).iterdir())
    )


# ========================================================================= #
#                                                                           #
#                               Subagents                                   #
#                                                                           #
# ========================================================================= #


def planner_subagent(
    objective: str, config: tern_config.Config, tern_dir: pathlib.Path
) -> str:
    tools = [web_fetch, read_file, list_files]
    model = tern_models.get_model(config, "planner").bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "planner")),
        lc_msg.HumanMessage(content=objective),
    ]

    max_iter = config.max_iterations.get("planner") or config.max_iterations["default"]
    response: object = None

    for _ in range(max_iter):
        response = model.invoke(messages)  # ty: ignore[invalid-argument-type]
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            return _extract_content(response)

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

    return _extract_content(response)


def maker_subagent(
    plan: str, config: tern_config.Config, tern_dir: pathlib.Path
) -> list[str]:
    return []


def checker_subagent(
    qa_output: str,
    file_contents: str,
    config: tern_config.Config,
    tern_dir: pathlib.Path,
) -> list[str]:
    return []


def summarizer_subagent(
    state: dict, config: tern_config.Config, tern_dir: pathlib.Path
) -> str:
    return ""


# ========================================================================= #
#                                                                           #
#                               Tool nodes                                  #
#                                                                           #
# ========================================================================= #


def dep_check_node(config: tern_config.Config, tern_dir: pathlib.Path) -> list[str]:
    return []


def qa_runner_node(config: tern_config.Config, tern_dir: pathlib.Path) -> str:
    return ""
