import argparse

import langchain.messages as lc_msg

import tern.agent as tern_agent

# ========================================================================= #
#                                                                           #
#                                   Main                                    #
#                                                                           #
# ========================================================================= #


def invoke_agent() -> None:
    agent = tern_agent.build_agent()

    state = {
        "objective": None,
        "plan": None,
        "plan_approved": None,
        "new_deps": [],
        "deps_approved": None,
        "qa_output": None,
        "issues": [],
        "need_handoff": False,
        "messages": [lc_msg.HumanMessage(content="Starting workflow")],
    }
    res = agent.invoke(state)

    print(res)


# ========================================================================= #
#                                                                           #
#                                   CLI                                     #
#                                                                           #
# ========================================================================= #


def get_cli_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # TODO

    return parser
