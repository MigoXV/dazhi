import datetime
import logging
import os

from openai.types.realtime import (
    AudioTranscription,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
    RealtimeFunctionTool,
)

from dazhi.inferencers.realtime.config import (
    RealtimeConfig,
    RealtimeConnectionConfig,
    RealtimeSessionConfig,
)
from dazhi.ui.chatbot import StreamChatbot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


async def get_current_time(args):
    return datetime.datetime.now().strftime("%H:%M:%S")


async def get_weather(args):
    return f"{args['city']} 当前温度为零下5度，大雪。"


def get_tool_executors():
    return {
        "get_current_time": get_current_time,
        "get_weather": get_weather,
    }


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
    chatbot = StreamChatbot(
        realtime_config=config, tools=get_tools(), tool_executors=get_tool_executors()
    )
    # chatbot = StreamChatbot(realtime_config=config)
    chatbot.launch()
