import pathlib

import tern.config as tern_config
import tern.subagents as subagents


def make_config() -> tern_config.Config:
    return tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6"},
        checker_tools=[],
        max_iterations={"default": 20},
    )


def test_planner_subagent_returns_str(tmp_path: pathlib.Path):
    assert isinstance(
        subagents.planner_subagent("build a model", make_config(), tmp_path), str
    )


def test_maker_subagent_returns_list(tmp_path: pathlib.Path):
    assert isinstance(subagents.maker_subagent("step 1", make_config(), tmp_path), list)


def test_checker_subagent_returns_list(tmp_path: pathlib.Path):
    result = subagents.checker_subagent("", "", make_config(), tmp_path)
    assert isinstance(result, list)


def test_summarizer_subagent_returns_str(tmp_path: pathlib.Path):
    assert isinstance(subagents.summarizer_subagent({}, make_config(), tmp_path), str)


def test_dep_check_node_returns_list(tmp_path: pathlib.Path):
    assert isinstance(subagents.dep_check_node(make_config(), tmp_path), list)


def test_qa_runner_node_returns_str(tmp_path: pathlib.Path):
    assert isinstance(subagents.qa_runner_node(make_config(), tmp_path), str)
