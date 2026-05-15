import operator
import typing as T

import langchain.messages as lc_msg
import langgraph.graph as lg_graph
import langgraph.graph.state as lg_state

import tern.subagents as tern_agents

# ========================================================================= #
#                                                                           #
#                               States                                      #
#                                                                           #
# ========================================================================= #


class AgentState(T.TypedDict):
    messages: T.Annotated[list[lc_msg.AnyMessage], operator.add]
    to_stop: bool


# ========================================================================= #
#                                                                           #
#                               Nodes                                       #
#                                                                           #
# ========================================================================= #


def planner_node(state: AgentState) -> dict:
    return {
        "messages": tern_agents.planner_subagent() + state["messages"],
        "to_stop": False,
    }


def evaluator_node(state: AgentState) -> dict:
    return {
        "messages": tern_agents.evaluator_subagent() + state["messages"],
        "to_stop": True,
    }


# ========================================================================= #
#                                                                           #
#                               Graph                                       #
#                                                                           #
# ========================================================================= #


def build_agent() -> lg_state.CompiledStateGraph[T.Any, T.Any, T.Any, T.Any]:
    graph = lg_graph.StateGraph(AgentState)  # ty: ignore[invalid-argument-type]

    graph.add_node("planner", planner_node)
    graph.add_node("evaluator", evaluator_node)

    graph.add_edge(lg_graph.START, "planner")
    graph.add_edge("planner", "evaluator")
    graph.add_edge("evaluator", lg_graph.END)

    return graph.compile()
