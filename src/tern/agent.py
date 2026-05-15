import operator
import typing as T

import langchain.messages as lc_msg
import langgraph.graph as lg_graph
import langgraph.graph.state as lg_state

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
    messages: T.Annotated[list[lc_msg.AnyMessage], operator.add]


# ========================================================================= #
#                                                                           #
#                               Nodes                                       #
#                                                                           #
# ========================================================================= #


def user_node(state: AgentState) -> dict:
    return {}


def planner_node(state: AgentState) -> dict:
    plan = tern_subagents.planner_subagent(state["objective"] or "")
    return {"plan": plan, "plan_approved": None}


def maker_node(state: AgentState) -> dict:
    tern_subagents.maker_subagent(state["plan"] or "")
    return {}


def dep_check_graph_node(state: AgentState) -> dict:
    new_deps = tern_subagents.dep_check_node()
    return {"new_deps": new_deps}


def qa_runner_graph_node(state: AgentState) -> dict:
    qa_output = tern_subagents.qa_runner_node()
    return {"qa_output": qa_output}


def checker_graph_node(state: AgentState) -> dict:
    issues = tern_subagents.checker_subagent(state["qa_output"] or "")
    return {"issues": issues}


def summarizer_graph_node(state: AgentState) -> dict:
    tern_subagents.summarizer_subagent(dict(state))
    return {}


# ========================================================================= #
#                                                                           #
#                               Routing                                     #
#                                                                           #
# ========================================================================= #


def route_from_user(state: AgentState) -> str:
    if not state["objective"]:
        return "summarizer" if state["need_handoff"] else lg_graph.END

    # Cycle complete: qa_runner and checker both ran with no issues
    if state["qa_output"] is not None and not state["issues"]:
        return "user"

    if state["plan_approved"] is not True:
        return "planner"

    if state["new_deps"] and state["deps_approved"] is None:
        return "user"

    if state["deps_approved"] is True:
        return "qa_runner"

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


def build_agent() -> lg_state.CompiledStateGraph[T.Any, T.Any, T.Any, T.Any]:
    graph = lg_graph.StateGraph(AgentState)  # ty: ignore[invalid-argument-type]

    graph.add_node("user", user_node)
    graph.add_node("planner", planner_node)
    graph.add_node("maker", maker_node)
    graph.add_node("dep_check", dep_check_graph_node)
    graph.add_node("qa_runner", qa_runner_graph_node)
    graph.add_node("checker", checker_graph_node)
    graph.add_node("summarizer", summarizer_graph_node)

    graph.add_edge(lg_graph.START, "user")
    graph.add_conditional_edges("user", route_from_user)
    graph.add_edge("planner", "user")
    graph.add_edge("maker", "dep_check")
    graph.add_conditional_edges("dep_check", route_from_dep_check)
    graph.add_edge("qa_runner", "checker")
    graph.add_conditional_edges("checker", route_from_checker)
    graph.add_edge("summarizer", lg_graph.END)

    return graph.compile(interrupt_before=["user"])
