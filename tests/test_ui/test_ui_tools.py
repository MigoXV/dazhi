#!/usr/bin/env python3
"""
麦当劳 MCP 智能助手 Demo - UI 模式
通过 Gradio UI 与 MCP 工具交互
"""
import asyncio
import logging
import os

import dotenv
from openai.types.realtime import (
    AudioTranscription,
    RealtimeAudioConfig,
    RealtimeAudioConfigInput,
)

from dazhi.inferencers.realtime.config import (
    RealtimeConfig,
    RealtimeConnectionConfig,
    RealtimeSessionConfig,
)
from dazhi.mcp_adaptors.config import MCPConfig
from dazhi.mcp_adaptors.mcp_client import MCPClient
from dazhi.ui.chatbot import StreamChatbot

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MCD_MCP_URL = "https://mcp.mcd.cn/mcp-servers/mcd-mcp"

# 全局 MCP 客户端（延迟初始化）
mcp_client: MCPClient | None = None
mcp_initialized = asyncio.Event()


async def ensure_mcp_connected():
    """确保 MCP 客户端已连接（在 Gradio 事件循环中调用）"""
    global mcp_client
    if mcp_client is None:
        mcp_config = MCPConfig(mcp_url=MCD_MCP_URL)
        mcp_client = MCPClient(mcp_config)
        await mcp_client.connect()
        mcp_initialized.set()
    return mcp_client


def create_tool_executor(tool_name: str):
    """为指定工具创建异步执行器"""

    async def executor(arguments: dict) -> str:
        print(f"\n🔧 调用工具: {tool_name}")
        print(f"   参数: {arguments}")
        try:
            # 确保 MCP 已连接
            client = await ensure_mcp_connected()
            result = await client.call_tool(tool_name, arguments)
            print(
                f"📋 工具结果: {result[:200]}..."
                if len(result) > 200
                else f"📋 工具结果: {result}"
            )
            return result
        except Exception as exc:
            error_msg = f"工具调用失败: {exc}"
            print(f"❌ {error_msg}")
            return error_msg

    return executor


def main():
    print("=" * 50)
    print("🍔 麦当劳 MCP 智能助手 (UI 模式)")
    print("=" * 50)

    # 同步获取工具列表（需要先连接一次）
    mcp_config = MCPConfig(mcp_url=MCD_MCP_URL)
    temp_client = MCPClient(mcp_config)

    async def get_tools():
        await temp_client.connect()
        tools = temp_client.get_tools_for_realtime()
        await temp_client.disconnect()
        return tools

    tools = asyncio.run(get_tools())
    print(f"\n📋 可用工具: {[t.name for t in tools]}")

    # 为每个工具创建对应的执行器
    tool_executors = {tool.name: create_tool_executor(tool.name) for tool in tools}

    # 配置 Realtime
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
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
    print("\nRealtimeConfig:\n", config)

    # 创建并启动 Chatbot
    chatbot = StreamChatbot(
        realtime_config=config,
        tools=tools,
        tool_executors=tool_executors,
    )

    chatbot.launch()


if __name__ == "__main__":
    main()
