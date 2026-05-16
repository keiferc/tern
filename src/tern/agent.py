import operator
import pathlib
import typing as T

import langchain.messages as lc_msg
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
    messages: T.Annotated[list[lc_msg.AnyMessage], operator.add]


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
    plan = tern_subagents.planner_subagent(state["objective"] or "", config, tern_dir)
    return {"plan": plan, "plan_approved": None}


def maker_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    files = tern_subagents.maker_subagent(state["plan"] or "", config, tern_dir)
    return {"written_files": files}


def dep_check_graph_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    new_deps = tern_subagents.dep_check_node(config, tern_dir)
    return {"new_deps": new_deps}


def qa_runner_graph_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    qa_output = tern_subagents.qa_runner_node(config, tern_dir)
    return {"qa_output": qa_output}


def checker_graph_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    file_contents = ""
    for path_str in state["written_files"]:
        try:
            content = pathlib.Path(path_str).read_text()
            file_contents += f"=== {path_str} ===\n{content}\n"
        except FileNotFoundError:
            pass
    issues = tern_subagents.checker_subagent(
        state["qa_output"] or "", file_contents, config, tern_dir
    )
    return {"issues": issues}


def summarizer_graph_node(
    state: AgentState, config: tern_config.Config, tern_dir: pathlib.Path
) -> dict:
    doc = tern_subagents.summarizer_subagent(dict(state), config, tern_dir)
    if doc:
        pathlib.Path.cwd().joinpath("HANDOFF.md").write_text(doc)
    return {}


# ========================================================================= #
#                                                                           #
#                               Routing                                     #
#                                                                           #
# ========================================================================= #


def route_from_user(state: AgentState) -> str:
    if not state["objective"]:
        return "summarizer" if state["need_handoff"] else lg_graph.END

    if state["plan_approved"] is not True:
        return "planner"

    # Cycle complete: full pipeline ran with no issues.
    # TODO: REPL must clear qa_output/issues/deps_approved when starting a
    # new objective so stale state from the previous cycle does not trigger this.
    if state["qa_output"] is not None and not state["issues"]:
        return "user"

    if state["new_deps"] and state["deps_approved"] is None:
        return "user"

    if state["deps_approved"] is True:
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
        return "maker"
    return "user"


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
    graph.add_node(
        "dep_check", lambda state: dep_check_graph_node(state, config, tern_dir)
    )
    graph.add_node(
        "qa_runner", lambda state: qa_runner_graph_node(state, config, tern_dir)
    )
    graph.add_node("checker", lambda state: checker_graph_node(state, config, tern_dir))
    graph.add_node(
        "summarizer", lambda state: summarizer_graph_node(state, config, tern_dir)
    )

    graph.add_edge(lg_graph.START, "user")
    graph.add_conditional_edges("user", route_from_user)
    graph.add_edge("planner", "user")
    graph.add_edge("maker", "dep_check")
    graph.add_conditional_edges("dep_check", route_from_dep_check)
    graph.add_edge("qa_runner", "checker")
    graph.add_conditional_edges("checker", route_from_checker)
    graph.add_edge("summarizer", lg_graph.END)

    return graph.compile(interrupt_before=["user"], checkpointer=checkpointer)
