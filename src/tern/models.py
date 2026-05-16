import langchain.chat_models as lc_chat

import tern.config as tern_config


def get_model(config: tern_config.Config, agent: str) -> lc_chat.BaseChatModel:
    model_string = config.models.get(agent) or config.models["default"]

    if ":" not in model_string:
        raise ValueError(
            f"model string must be in 'provider:model-name' format, got: {model_string!r}"
        )

    provider, model_name = model_string.split(":", 1)
    if not provider:
        raise ValueError(f"model string has empty provider segment: {model_string!r}")
    if not model_name:
        raise ValueError(f"model string has empty model-name segment: {model_string!r}")

    return lc_chat.init_chat_model(model_string)
