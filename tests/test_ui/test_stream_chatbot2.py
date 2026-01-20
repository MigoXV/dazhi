import datetime
import queue
import random
import threading
import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from gradio import ChatMessage

from dazhi.ui.chatbot import StreamChatbot


def stream_text(
    text: str, min_delay: float = 0.01, max_delay: float = 0.04
) -> Iterable[str]:
    for character in text:
        time.sleep(random.uniform(min_delay, max_delay))
        yield character


def stub_reply(history: List[Dict]) -> Generator[str, None, None]:
    """生成 stub 回复，用于驱动 StreamChatbot.handle_bot 的流式输出"""
    last_user_message = ""
    if history:
        last_user_message = history[-1]["content"]
        response = "今天的天气真好" * 30 + f"，你刚才说的是：{last_user_message}"
    else:
        response = "未收到用户消息。"
    return stream_text(response)


if __name__ == "__main__":
    chatbot = StreamChatbot(
        bot_stream_handler=stub_reply,
    )
    chatbot.launch()
