import datetime
import logging
import os
import queue
import random
import threading
import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from gradio import ChatMessage
from openai.types.realtime import RealtimeFunctionTool

from dazhi.inferencers.realtime.config import (
    RealtimeConfig,
    RealtimeConnectionConfig,
    RealtimeSessionConfig,
)
from dazhi.ui.chatbot import StreamChatbot
from openai.types.realtime import (
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    AudioTranscription,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def stream_text(
    text: str, min_delay: float = 0.01, max_delay: float = 0.04
) -> Iterable[str]:
    for character in text:
        time.sleep(random.uniform(min_delay, max_delay))
        yield character


def get_tools():
    tool1 = RealtimeFunctionTool(
        name="get_current_time",
        description="获取当前的时间，格式为 HH:MM:SS",
        type="function",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
    tool2 = RealtimeFunctionTool(
        name="get_weather",
        description="获取指定城市的当前天气情况",
        type="function",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称",
                },
            },
            "required": ["city"],
        },
    )
    return [tool1, tool2]


async def stub_function_callback(function_name: str, event) -> str | None:
    """
    Stub 回调函数 - 无论调用什么工具，都返回 25 摄氏度

    Args:
        function_name: 函数名（从 ConversationCreatedEvent 中提取）
        event: function call 完成事件，包含 arguments（参数 JSON）、call_id

    Returns:
        工具调用结果字符串
    """

    # 无论什么工具调用，都返回 25 摄氏度
    return "当前温度为 25 摄氏度，天气晴朗"


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
    model = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
    config = RealtimeConfig(
        connection=RealtimeConnectionConfig(model=model),
        session=RealtimeSessionConfig(
            output_modalities=["text"],
            audio=RealtimeAudioConfig(
                input=RealtimeAudioConfigInput(
                    transcription=AudioTranscription(model="gpt-4o-transcribe")
                )
            ),
        ),
    )
    print("RealtimeConfig:\n\n", config)
    chatbot = StreamChatbot(realtime_config=config, tools=get_tools())
    # chatbot = StreamChatbot(realtime_config=config)
    chatbot.launch()
