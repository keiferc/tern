import langchain.chat_models as lc_chat

import tern.config as tern_config


def get_model(config: tern_config.Config, agent: str) -> lc_chat.BaseChatModel:
    return lc_chat.init_chat_model(config.models[agent])
