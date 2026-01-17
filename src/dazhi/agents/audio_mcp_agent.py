"""语音 MCP Agent - 结合语音识别、LLM 推理与 MCP 工具调用"""

import asyncio
import json

from dazhi.handlers.voice_mcp import VoiceMCPEventHandler
from dazhi.inferencers.llm.inferencer import LLMInferencer
from dazhi.inferencers.llm.session import LLMChatSession
from dazhi.inferencers.realtime.inferencer import RealtimeConfig, RealtimeInferencer
from dazhi.mcp_adaptors.config import LLMConfig, MCPConfig
from dazhi.mcp_adaptors.mcp_client import MCPClient


class VoiceMCPAgent:
    """语音 + MCP 推理器

    结合实时语音识别和 MCP 工具调用，实现语音控制的智能助手。

    Example:
        ```python
        mcp_config = MCPConfig(mcp_url="https://mcp.mcd.cn/mcp-servers/mcd-mcp")
        agent = VoiceMCPAgent(mcp_config=mcp_config)
        await agent.run()
        ```
    """

    def __init__(
        self,
        mcp_config: MCPConfig,
        llm_config: LLMConfig | None = None,
        realtime_config: RealtimeConfig | None = None,
    ):
        self.mcp_config = mcp_config
        self.llm_config = llm_config or LLMConfig()
        self.realtime_config = realtime_config or RealtimeConfig()

        self.mcp_client = MCPClient(mcp_config)
        self.llm_inferencer = LLMInferencer(llm_config)
        self.realtime_inferencer: RealtimeInferencer | None = None
        self.llm_session: LLMChatSession | None = None

        self.is_running = False
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()

    async def _process_transcripts(self) -> None:
        """处理语音转写结果队列"""
        while self.is_running:
            try:
                # 等待语音输入
                transcript = await asyncio.wait_for(
                    self._transcript_queue.get(), timeout=0.5
                )

                # 检查是否退出
                if transcript in ["退出", "结束", "停止", "exit", "quit"]:
                    print("\n👋 收到退出指令，正在停止...")
                    self.is_running = False
                    break

                # 处理语音输入
                tools = self.mcp_client.get_tools_for_openai()
                if not self.llm_session:
                    raise RuntimeError("LLM session 未初始化")

                assistant_message = await self.llm_inferencer.process_user_input(
                    self.llm_session, transcript, tools=tools
                )

                tool_calls = assistant_message.get("tool_calls", [])
                if tool_calls:
                    await self._handle_tool_calls(tool_calls)
                    await self.llm_inferencer.continue_assistant(self.llm_session)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"\n❌ 处理转写结果出错: {e}")

    async def _handle_tool_calls(self, tool_calls: list[dict]) -> None:
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = (
                json.loads(tool_call["function"]["arguments"])
                if tool_call["function"]["arguments"]
                else {}
            )

            print(f"🔧 调用工具: {tool_name}({tool_args})")

            try:
                result = await self.mcp_client.call_tool(tool_name, tool_args)
                print(
                    f"📋 工具结果: {result[:200]}..."
                    if len(result) > 200
                    else f"📋 工具结果: {result}"
                )
            except Exception as e:
                result = f"工具调用失败: {e}"
                print(f"❌ {result}")

            if self.llm_session:
                self.llm_session.add_tool_result(tool_call["id"], result)

    async def run(self) -> None:
        """运行语音 MCP 模式"""
        print("=" * 50)
        print("🍔 麦当劳 MCP 智能助手 (语音模式)")
        print("=" * 50)

        self.is_running = True

        # 初始化 MCP 推理器
        await self.llm_inferencer._init_llm_client()
        await self.mcp_client.connect()
        self.llm_session = LLMChatSession(
            system_prompt=self.llm_inferencer.llm_config.system_prompt
        )
        self.llm_inferencer.is_running = True

        print("\n🎤 语音模式启动，请说话...")
        print("   说 '退出' 或 '结束' 停止\n")

        # 创建事件处理器
        event_handler = VoiceMCPEventHandler(self._transcript_queue)

        # 创建实时推理器
        self.realtime_inferencer = RealtimeInferencer(
            config=self.realtime_config,
            event_handler=event_handler,
        )

        try:
            # 并行运行语音识别和转写处理
            await asyncio.gather(
                self.realtime_inferencer.run(enable_audio_playback=False),
                self._process_transcripts(),
            )
        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，正在停止...")
        finally:
            self.is_running = False
            if self.realtime_inferencer:
                await self.realtime_inferencer.stop()
            await self.mcp_client.disconnect()
            print("✓ 所有连接已关闭")

    async def __aenter__(self) -> "VoiceMCPAgent":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.is_running = False
        if self.realtime_inferencer:
            await self.realtime_inferencer.stop()
        await self.mcp_client.disconnect()
