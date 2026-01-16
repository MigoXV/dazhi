#!/usr/bin/env python3
"""
麦当劳 MCP 智能助手 Demo - 语音模式
通过语音识别与 MCP 工具交互
"""
import asyncio

import dotenv

from dazhi.inferencers.mcp.inferencer import (
    LLMConfig,
    MCPConfig,
    VoiceMCPInferencer,
)
from dazhi.inferencers.realtime.inferencer import RealtimeConfig

dotenv.load_dotenv()

MCD_MCP_URL = "https://mcp.mcd.cn/mcp-servers/mcd-mcp"

SYSTEM_PROMPT = """你是麦当劳智能助手，可以帮助用户：
- 查询当前时间 (now-time-info)
- 查询活动日历 (campaign-calender)
- 查询可用优惠券 (available-coupons)
- 以及其他麦当劳相关服务

请用中文回答，简洁明了。当需要查询信息时，请调用相应的工具。"""


async def main():
    mcp_config = MCPConfig(mcp_url=MCD_MCP_URL)
    llm_config = LLMConfig(model="qwen3:8B", system_prompt=SYSTEM_PROMPT)
    realtime_config = RealtimeConfig(model="transcribe", output_modalities=["text"])

    inferencer = VoiceMCPInferencer(
        mcp_config=mcp_config,
        llm_config=llm_config,
        realtime_config=realtime_config,
    )
    await inferencer.run()


if __name__ == "__main__":
    asyncio.run(main())
