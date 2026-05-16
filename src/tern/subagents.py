import pathlib

import tern.config as tern_config

# ========================================================================= #
#                                                                           #
#                               Subagents                                   #
#                                                                           #
# ========================================================================= #


def planner_subagent(
    objective: str, config: tern_config.Config, tern_dir: pathlib.Path
) -> str:
    return ""


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
