import fnmatch
import pathlib
import typing as T
import urllib.parse
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
    path = tern_dir / "CONSTITUTION.md"
    try:
        constitution = path.read_text()
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


def _safe_resolve(path_str: str) -> pathlib.Path:
    cwd = pathlib.Path.cwd().resolve()
    resolved = (cwd / path_str).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ValueError(f"path outside working directory: {path_str!r}")
    return resolved


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
    if response is None:
        raise ValueError(
            f"{agent_name} produced no response: max_iterations is {max_iter}"
        )
    if getattr(response, "tool_calls", []):
        raise RuntimeError(
            f"{agent_name}: max_iterations exhausted with pending tool calls"
        )
    return response


_SENSITIVE_FILE_ALLOWLIST = frozenset({".env.example"})
_SENSITIVE_FILE_PATTERNS = (
    "*.env*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    "*credentials*",
    "*secret*",
    "*_token*",
)


# ========================================================================= #
#                                                                           #
#                               Tools                                       #
#                                                                           #
# ========================================================================= #


@lc_tools.tool
def web_fetch(url: str) -> str:
    """Fetch the text content of a URL."""
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        return f"Error: web_fetch only supports http/https, got scheme {scheme!r}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            text = resp.read().decode("utf-8", errors="replace")
            if len(text) > 20000:
                return text[:20000] + "\n[... truncated]"
            return text
    except Exception as exc:
        return f"Error fetching {url!r}: {exc}"


@lc_tools.tool
def read_file(path: str) -> str:
    """Read a file within the project working directory and return its content."""
    resolved = _safe_resolve(path)
    name = resolved.name
    if name not in _SENSITIVE_FILE_ALLOWLIST and any(
        fnmatch.fnmatch(name, pat) for pat in _SENSITIVE_FILE_PATTERNS
    ):
        raise ValueError(f"read_file: {name!r} matches a sensitive-file pattern")
    return resolved.read_text()


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
    objective: str,
    config: tern_config.Config,
    tern_dir: pathlib.Path,
    *,
    prior_plan: str | None = None,
) -> str:
    tools = [web_fetch, read_file, list_files]
    model = tern_models.get_model(config, "planner").bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "planner")),
        lc_msg.HumanMessage(content=objective),
    ]
    if prior_plan:
        messages.append(lc_msg.AIMessage(content=prior_plan))
    _val = config.max_iterations.get("planner")
    max_iter = config.max_iterations["default"] if _val is None else _val
    response = _react_loop(model, tool_map, messages, max_iter, "planner_subagent")
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
    tools = [web_fetch, read_file, list_files]
    model = tern_models.get_model(config, "checker").bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    human_message = (
        "The following files were written by the maker. These are your primary review target.\n"
        "Use tools only to read additional project files or verify documentation — "
        "do not re-read files already provided here.\n"
        "\n"
        "## QA Tool Output\n"
        f"{qa_output}\n"
        "\n"
        "## Written Files\n"
        f"{file_contents}\n"
        "\n"
        "Report each issue on its own line. "
        "Output only issues — no preamble, no summary, no explanation."
    )

    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "checker")),
        lc_msg.HumanMessage(content=human_message),
    ]
    _val = config.max_iterations.get("checker")
    max_iter = config.max_iterations["default"] if _val is None else _val
    response = _react_loop(model, tool_map, messages, max_iter, "checker_subagent")
    content = _extract_content(response)
    return [line for line in (ln.strip() for ln in content.splitlines()) if line]


def summarizer_subagent(
    state: dict, config: tern_config.Config, tern_dir: pathlib.Path
) -> str:
    parts: list[str] = []

    if state.get("objective"):
        parts.append(f"## Objective\n{state['objective']}")
    if state.get("plan"):
        parts.append(f"## Plan\n{state['plan']}")
    if state.get("written_files"):
        parts.append("## Written Files\n" + "\n".join(state["written_files"]))

    for msg in state.get("messages", []):
        if isinstance(msg, lc_msg.HumanMessage):
            text = _extract_content(msg)
            if text:
                parts.append(f"## User Message\n{text}")

    context = "\n\n".join(parts)
    if not context:
        return ""

    model = tern_models.get_model(config, "summarizer")
    messages: list[object] = [
        lc_msg.SystemMessage(content=_build_system_prompt(tern_dir, "summarizer")),
        lc_msg.HumanMessage(content=context),
    ]
    response = model.invoke(messages)  # ty: ignore[invalid-argument-type]
    return _extract_content(response)


# ========================================================================= #
#                                                                           #
#                               Tool nodes                                  #
#                                                                           #
# ========================================================================= #


def dep_check_node(config: tern_config.Config, tern_dir: pathlib.Path) -> list[str]:
    return []


def qa_runner_node(config: tern_config.Config, tern_dir: pathlib.Path) -> str:
    return ""
