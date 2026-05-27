import operator
import pathlib
import re
import shlex
import subprocess
import tomllib
import typing as T

import langgraph.graph as lg_graph
import langgraph.graph.state as lg_state

import tern.config as tern_config
import tern.subagents as tern_subagents

# ========================================================================= #
#                                                                           #
#                               State                                       #
#                                                                           #
# ========================================================================= #


class AgentState(T.TypedDict):
    objective: str | None
    plan: str | None
    plan_approved: bool | None
    new_deps: list[str]
    deps_approved: bool | None
    qa_output: str | None
    issues: list[str]
    need_handoff: bool
    written_files: list[str]
    feedback: list[str]
    maker_checker_cycles: int
    milestones: T.Annotated[list[str], operator.add]


INITIAL_STATE: dict = {
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
}


# ========================================================================= #
#                                                                           #
#                               Nodes                                       #
#                                                                           #
# ========================================================================= #


def user_node(state: AgentState) -> dict:
    return {}


def planner_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    print("planning…", flush=True)
    plan = tern_subagents.planner_subagent(
        state["objective"],  # ty: ignore[invalid-argument-type]
        config,
        tern_dir,
        prior_plan=state["plan"],
        issues=state["issues"],
        feedback=state["feedback"],
    )
    return {
        "plan": plan,
        "plan_approved": None,
        "qa_output": None,
        "issues": [],
        "feedback": [],
        "maker_checker_cycles": 0,
    }


def maker_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    print("implementing…", flush=True)
    files = tern_subagents.maker_subagent(
        state["objective"],  # ty: ignore[invalid-argument-type]
        state["plan"],  # ty: ignore[invalid-argument-type]
        config,
        tern_dir,
        issues=state["issues"],
        feedback=state["feedback"],
    )
    return {
        "written_files": files,
        "maker_checker_cycles": state["maker_checker_cycles"] + 1,
    }


def dep_check_node(state: AgentState) -> dict:
    print("checking dependencies…", flush=True)
    cwd = pathlib.Path.cwd()
    pyproject = tomllib.loads((cwd / "pyproject.toml").read_text(encoding="utf-8"))
    raw_deps = pyproject.get("project", {}).get("dependencies", [])
    pyproject_names = {
        _normalize_pkg(re.split(r"[\s\[;!<>=]", dep)[0]) for dep in raw_deps
    }
    lock = tomllib.loads((cwd / "uv.lock").read_text(encoding="utf-8"))
    lock_names = {_normalize_pkg(pkg["name"]) for pkg in lock.get("package", [])}
    return {"new_deps": sorted(pyproject_names - lock_names)}


def qa_runner_node(config: tern_config.Config) -> dict:
    print("running QA…", flush=True)
    parts_list = []
    for cmd in config.checker_tools:
        parts = shlex.split(cmd)
        if not parts:
            continue
        parts_list.append((cmd, parts))
    output = ""
    for cmd, parts in parts_list:
        try:
            result = subprocess.run(parts, shell=False, capture_output=True, text=True)
            output += f"$ {cmd}\n{result.stdout}{result.stderr}\n"
        except OSError as e:
            output += f"$ {cmd}\nError: {e}\n"
    return {"qa_output": output}


def checker_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    print("reviewing…", flush=True)
    file_contents = ""
    cwd = pathlib.Path.cwd().resolve()
    for path_str in state["written_files"]:
        try:
            resolved = pathlib.Path(path_str).resolve()
            rel = resolved.relative_to(cwd)
            content = resolved.read_text(encoding="utf-8")
            file_contents += f"=== {rel} ===\n{content}\n"
        except FileNotFoundError, ValueError:
            pass
    issues = tern_subagents.checker_subagent(
        state["qa_output"],  # ty: ignore[invalid-argument-type]
        file_contents,
        config,
        tern_dir,
    )
    if not issues:
        return {
            "issues": [],
            "feedback": [],
            "maker_checker_cycles": 0,
            "milestones": [state["plan"]],
        }
    if state["maker_checker_cycles"] >= config.max_iterations["maker_checker_cycles"]:
        return {"issues": issues, "plan_approved": None, "feedback": []}
    return {"issues": issues, "feedback": []}


def summarizer_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    print("generating handoff…", flush=True)
    doc = tern_subagents.summarizer_subagent(dict(state), config, tern_dir)
    if doc:
        pathlib.Path.cwd().joinpath("HANDOFF.md").write_text(doc, encoding="utf-8")
    return {}


# ========================================================================= #
#                                                                           #
#                               Routing                                     #
#                                                                           #
# ========================================================================= #


def route_from_user(state: AgentState) -> str:
    if not state["objective"]:
        return lg_graph.END

    if state["need_handoff"]:
        return "summarizer"

    if state["plan_approved"] is not True:
        return "planner"

    if (
        state["qa_output"] is not None
        and not state["issues"]
        and not (state["new_deps"] and state["deps_approved"] is None)
    ):
        return "user"

    if state["new_deps"] and state["deps_approved"] is None:
        return "user"

    if state["new_deps"] and state["deps_approved"] is True:
        return "qa_runner"

    # deps_approved=False: user rejected the proposed deps; send maker back
    # to rework the implementation without adding them.
    return "maker"


def route_from_dep_check(state: AgentState) -> str:
    if state["new_deps"]:
        return "user"
    return "qa_runner"


def route_from_checker(state: AgentState) -> str:
    if state["issues"]:
        if state["plan_approved"] is None:
            return "user"
        return "maker"
    return "user"


# ========================================================================= #
#                                                                           #
#                               Helpers                                     #
#                                                                           #
# ========================================================================= #


def _normalize_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ========================================================================= #
#                                                                           #
#                               Graph                                       #
#                                                                           #
# ========================================================================= #


def build_agent(
    config: tern_config.Config,
    tern_dir: pathlib.Path,
    checkpointer: T.Any = None,
) -> lg_state.CompiledStateGraph[T.Any, T.Any, T.Any, T.Any]:
    graph = lg_graph.StateGraph(AgentState)  # ty: ignore[invalid-argument-type]

    graph.add_node("user", user_node)
    graph.add_node("planner", lambda state: planner_node(state, config, tern_dir))
    graph.add_node("maker", lambda state: maker_node(state, config, tern_dir))
    graph.add_node("dep_check", dep_check_node)
    graph.add_node("qa_runner", lambda state: qa_runner_node(config))  # noqa: ARG005
    graph.add_node("checker", lambda state: checker_node(state, config, tern_dir))
    graph.add_node("summarizer", lambda state: summarizer_node(state, config, tern_dir))

    graph.add_edge(lg_graph.START, "user")
    graph.add_conditional_edges("user", route_from_user)
    graph.add_edge("planner", "user")
    graph.add_edge("maker", "dep_check")
    graph.add_conditional_edges("dep_check", route_from_dep_check)
    graph.add_edge("qa_runner", "checker")
    graph.add_conditional_edges("checker", route_from_checker)
    graph.add_edge("summarizer", lg_graph.END)

    return graph.compile(interrupt_before=["user"], checkpointer=checkpointer)
