# ========================================================================= #
#                                                                           #
#                               Subagents                                   #
#                                                                           #
# ========================================================================= #


def planner_subagent(objective: str) -> str:
    return ""


def maker_subagent(plan: str) -> None:
    pass


def checker_subagent(qa_output: str) -> list[str]:
    return []


def summarizer_subagent(state: dict) -> str:
    return ""


# ========================================================================= #
#                                                                           #
#                               Tool nodes                                  #
#                                                                           #
# ========================================================================= #


def dep_check_node() -> list[str]:
    return []


def qa_runner_node() -> str:
    return ""
