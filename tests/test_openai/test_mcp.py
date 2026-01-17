#!/usr/bin/env python3
"""
MCP + Realtime API 端到端测试
将 MCP 工具映射为 OpenAI Realtime 工具，实现语音+MCP工具调用
"""
import asyncio
import json
import os

import dotenv
from openai.types.realtime import (
    RealtimeFunctionTool,
    ResponseFunctionCallArgumentsDoneEvent,
)

from dazhi.handlers.default_event import DefaultEventHandler
from dazhi.inferencers.realtime.inferencer import RealtimeConfig, RealtimeInferencer
from dazhi.mcp_adaptors.config import MCPConfig
from dazhi.mcp_adaptors.mcp_client import MCPClient

dotenv.load_dotenv()

MCD_MCP_URL = "https://mcp.mcd.cn/mcp-servers/mcd-mcp"

# 全局 MCP 客户端引用（供回调函数使用）
_mcp_client: MCPClient | None = None


def mcp_tools_to_realtime_tools(mcp_client: MCPClient) -> list[RealtimeFunctionTool]:
    """将 MCP 工具转换为 OpenAI Realtime 工具格式"""
    realtime_tools = []
    for tool in mcp_client.tools:
        realtime_tool = RealtimeFunctionTool(
            name=tool.name,
            description=tool.description or "",
            parameters=(
                tool.inputSchema
                if tool.inputSchema
                else {"type": "object", "properties": {}, "required": []}
            ),
        )
        realtime_tools.append(realtime_tool)
    return realtime_tools


async def mcp_function_callback(
    function_name: str,
    event: ResponseFunctionCallArgumentsDoneEvent,
) -> str | None:
    """
    MCP 工具回调函数 - 调用 MCP 工具并返回结果

    Args:
        function_name: 函数名
        event: function call 完成事件，包含 arguments（参数 JSON）、call_id

    Returns:
        工具调用结果字符串
    """
    global _mcp_client

    print(f"\n[MCP] 工具调用: {function_name}")
    print(f"[MCP] 参数: {event.arguments}")
    print(f"[MCP] Call ID: {event.call_id}")

    if _mcp_client is None:
        return "MCP 客户端未初始化"

    try:
        # 解析参数
        arguments = json.loads(event.arguments) if event.arguments else {}

        # 调用 MCP 工具
        result = await _mcp_client.call_tool(function_name, arguments)
        print(
            f"[MCP] 工具结果: {result[:200]}..."
            if len(result) > 200
            else f"[MCP] 工具结果: {result}"
        )
        return result
    except Exception as e:
        error_msg = f"工具调用失败: {e}"
        print(f"[MCP] ❌ {error_msg}")
        return error_msg


async def main():
    global _mcp_client

    model = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
    print(f"Using model: {model}")

    # 初始化 MCP 客户端
    mcp_config = MCPConfig(mcp_url=MCD_MCP_URL)
    _mcp_client = MCPClient(mcp_config)

    try:
        # 连接 MCP 并获取工具
        await _mcp_client.connect()

        # 转换 MCP 工具为 Realtime 格式
        tools = mcp_tools_to_realtime_tools(_mcp_client)
        print(f"✓ 已加载 {len(tools)} 个 MCP 工具: {[t.name for t in tools]}")

        # 配置
        config = RealtimeConfig(
            model=model,
            output_modalities=["text"],
        )

        # 初始化事件处理器（注入 MCP 回调函数）
        handler = DefaultEventHandler(
            audio_player=None,
            on_function_call_done_callback=mcp_function_callback,
        )

        # 初始化推理器
        inferencer = RealtimeInferencer(
            config=config,
            event_handler=handler,
            tools=tools,
        )

        await inferencer.run()

    finally:
        # 确保 MCP 连接正确关闭
        if _mcp_client:
            await _mcp_client.disconnect()
            print("✓ MCP 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
