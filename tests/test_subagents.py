import pathlib

import tern.config as tern_config
import tern.subagents as subagents


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_tern_dir(tmp_path: pathlib.Path, override: str = "") -> pathlib.Path:
    (tmp_path / "CONSTITUTION.md").write_text("# Constitution\nrule 1")
    (tmp_path / "planner.md").write_text(override)
    return tmp_path


def test_build_system_prompt_constitution_only(tmp_path: pathlib.Path):
    tern_dir = _write_tern_dir(tmp_path, override="")
    result = subagents._build_system_prompt(tern_dir, "planner")
    assert result == "# Constitution\nrule 1"


def test_build_system_prompt_concatenates_non_empty_override(tmp_path: pathlib.Path):
    tern_dir = _write_tern_dir(tmp_path, override="You are a planner.")
    result = subagents._build_system_prompt(tern_dir, "planner")
    assert result == "# Constitution\nrule 1\n\nYou are a planner."


def test_build_system_prompt_whitespace_only_override_ignored(tmp_path: pathlib.Path):
    tern_dir = _write_tern_dir(tmp_path, override="   \n  ")
    result = subagents._build_system_prompt(tern_dir, "planner")
    assert result == "# Constitution\nrule 1"


# ── fixtures ──────────────────────────────────────────────────────────────────


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
