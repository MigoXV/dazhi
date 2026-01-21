#!/usr/bin/env python3
"""
Simplified Realtime API app - immediately starts audio streaming and receiving results.
使用 RealtimeInferencer 推理器的简化版本。
"""
import asyncio
import json
import os
import dotenv
from openai.types.realtime import RealtimeFunctionTool, ResponseFunctionCallArgumentsDoneEvent

from dazhi.codec import SDAudioRecorder
from dazhi.inferencers.realtime.inferencer import RealtimeConfig, RealtimeInferencer
from dazhi.handlers.default_event import DefaultEventHandler

dotenv.load_dotenv()


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


async def stub_function_callback(
    function_name: str,
    event: ResponseFunctionCallArgumentsDoneEvent,
) -> str | None:
    """
    Stub 回调函数 - 无论调用什么工具，都返回 25 摄氏度
    
    Args:
        function_name: 函数名（从 ConversationCreatedEvent 中提取）
        event: function call 完成事件，包含 arguments（参数 JSON）、call_id
    
    Returns:
        工具调用结果字符串
    """
    print(f"\n[DEBUG] 工具调用: {function_name}")
    print(f"[DEBUG] 参数: {event.arguments}")
    print(f"[DEBUG] Call ID: {event.call_id}")
    
    # 无论什么工具调用，都返回 25 摄氏度
    return "当前温度为 25 摄氏度，天气晴朗"


def main():
    model = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
    print(f"Using model: {model}")
    
    # 配置
    config = RealtimeConfig(
        model=model,
        output_modalities=["text"],
    )
    
    # 初始化事件处理器（注入回调函数）
    handler = DefaultEventHandler(
        audio_player=None,
        on_function_call_done_callback=stub_function_callback,
    )
    
    # 初始化推理器（依赖注入 handler）
    inferencer = RealtimeInferencer(
        config=config,
        event_handler=handler,
        tools=get_tools(),
    )
    
    recorder = SDAudioRecorder(
        channels=config.channels,
        sample_rate=config.sample_rate,
    )
    asyncio.run(inferencer.run(audio_recorder=recorder))


if __name__ == "__main__":
    main()
