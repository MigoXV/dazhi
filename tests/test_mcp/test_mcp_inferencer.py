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
from dazhi.mcp_adaptors.config import LLMConfig, MCPConfig

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
    llm_api_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.llmbase")
    llm_model = os.environ.get("LLM_MODEL", "qwen3:8B")
    llm_config = LLMConfig(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
        system_prompt=SYSTEM_PROMPT,
    )
    print("使用 LLM 配置：", llm_config)
    realtime_config = RealtimeConfig(model="transcrib", output_modalities=["text"])
    print("使用实时推理配置：", realtime_config)
    inferencer = VoiceMCPAgent(
        mcp_config=mcp_config,
        llm_config=llm_config,
        realtime_config=realtime_config,
    )
    await inferencer.run()


if __name__ == "__main__":
    asyncio.run(main())
