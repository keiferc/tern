import unittest.mock

import pytest

import tern.config as tern_config
import tern.models as tern_models


def make_config(**models: str) -> tern_config.Config:
    return tern_config.Config(
        models={"default": "anthropic:claude-sonnet-4-6", **models},
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


def test_get_model_unknown_agent_falls_back_to_default():
    config = make_config()
    with unittest.mock.patch("langchain.chat_models.init_chat_model") as mock_init:
        tern_models.get_model(config, "checker")
        mock_init.assert_called_once_with("anthropic:claude-sonnet-4-6")


def test_get_model_returns_init_chat_model_result():
    config = make_config()
    sentinel = unittest.mock.MagicMock()
    with unittest.mock.patch(
        "langchain.chat_models.init_chat_model", return_value=sentinel
    ):
        result = tern_models.get_model(config, "planner")
        assert result is sentinel


# ── validation ────────────────────────────────────────────────────────────────


def test_get_model_raises_on_missing_colon():
    config = make_config(planner="claudesonnet")
    with pytest.raises(ValueError, match="provider:model-name"):
        tern_models.get_model(config, "planner")


def test_get_model_raises_on_empty_provider():
    config = make_config(planner=":claude-sonnet-4-6")
    with pytest.raises(ValueError, match="empty provider"):
        tern_models.get_model(config, "planner")


def test_get_model_raises_on_empty_model_name():
    config = make_config(planner="anthropic:")
    with pytest.raises(ValueError, match="empty model-name"):
        tern_models.get_model(config, "planner")
