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

    messages = [lc_msg.HumanMessage(content="Starting workflow")]
    res = agent.invoke({"messages": messages})

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
