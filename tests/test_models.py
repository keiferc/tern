import unittest.mock

import tern.config as tern_config
import tern.models as tern_models


def make_config(**models: str) -> tern_config.Config:
    return tern_config.Config(
        models={"planner": "anthropic:claude-sonnet-4-6", **models},
        checker_tools=[],
        max_iterations={"default": 20},
    )


# ── provider dispatch ─────────────────────────────────────────────────────────


def test_get_model_calls_init_chat_model_with_full_string():
    config = make_config()
    with unittest.mock.patch("langchain.chat_models.init_chat_model") as mock_init:
        tern_models.get_model(config, "planner")
        mock_init.assert_called_once_with("anthropic:claude-sonnet-4-6")


def test_get_model_per_agent_key_overrides_default():
    config = make_config(maker="openai:gpt-4o")
    with unittest.mock.patch("langchain.chat_models.init_chat_model") as mock_init:
        tern_models.get_model(config, "maker")
        mock_init.assert_called_once_with("openai:gpt-4o")


def test_get_model_returns_init_chat_model_result():
    config = make_config()
    sentinel = unittest.mock.MagicMock()
    with unittest.mock.patch(
        "langchain.chat_models.init_chat_model", return_value=sentinel
    ):
        result = tern_models.get_model(config, "planner")
        assert result is sentinel
