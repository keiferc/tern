import tern.subagents as subagents


def test_planner_subagent_returns_str():
    assert isinstance(subagents.planner_subagent("build a model"), str)


def test_maker_subagent_returns_none():
    assert subagents.maker_subagent("step 1") is None


def test_checker_subagent_returns_list():
    result = subagents.checker_subagent("")
    assert isinstance(result, list)


def test_summarizer_subagent_returns_str():
    assert isinstance(subagents.summarizer_subagent({}), str)


def test_dep_check_node_returns_list():
    result = subagents.dep_check_node()
    assert isinstance(result, list)


def test_qa_runner_node_returns_str():
    assert isinstance(subagents.qa_runner_node(), str)
