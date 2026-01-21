#!/usr/bin/env python3
"""
麦当劳 MCP 智能助手 Demo - 语音模式
通过语音识别与 MCP 工具交互
"""
import asyncio
import os

import dotenv

from dazhi.agents.audio_mcp_agent import VoiceMCPAgent
from dazhi.inferencers.realtime.inferencer import RealtimeConfig
from dazhi.mcp_adaptors.config import MCPConfig

dotenv.load_dotenv()

MCD_MCP_URL = "https://mcp.mcd.cn/mcp-servers/mcd-mcp"

async def main():
    mcp_config = MCPConfig(mcp_url=MCD_MCP_URL)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    realtime_config = RealtimeConfig(model=model, output_modalities=["text"])
    print("使用实时推理配置：", realtime_config)
    inferencer = VoiceMCPAgent(
        mcp_config=mcp_config,
        realtime_config=realtime_config,
    )
    await inferencer.run()


if __name__ == "__main__":
    asyncio.run(main())
