"""语音 MCP Agent - 结合 Realtime 语音识别与 MCP 工具调用"""

import json

from openai.types.realtime import ResponseFunctionCallArgumentsDoneEvent

from dazhi.handlers.voice_mcp import VoiceMCPEventHandler
from dazhi.inferencers.realtime.inferencer import RealtimeConfig, RealtimeInferencer
from dazhi.mcp_adaptors.config import MCPConfig
from dazhi.mcp_adaptors.mcp_client import MCPClient


class VoiceMCPAgent:
    """语音 + MCP 推理器

    仅使用 Realtime API 完成语音识别与工具调用。

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
        realtime_config: RealtimeConfig | None = None,
    ):
        self.mcp_config = mcp_config
        self.realtime_config = realtime_config or RealtimeConfig()

        self.mcp_client = MCPClient(mcp_config)
        self.realtime_inferencer: RealtimeInferencer | None = None

        self.is_running = False

    async def _on_function_call_done(
        self,
        function_name: str,
        event: ResponseFunctionCallArgumentsDoneEvent,
    ) -> str | None:
        """Realtime 工具调用完成后的回调，转发到 MCP"""
        print(f"\n🔧 调用工具: {function_name}")
        print(f"   参数: {event.arguments}")
        try:
            arguments = json.loads(event.arguments) if event.arguments else {}
            result = await self.mcp_client.call_tool(function_name, arguments)
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

    async def run(self) -> None:
        """运行语音 MCP 模式"""
        print("=" * 50)
        print("🍔 麦当劳 MCP 智能助手 (语音模式)")
        print("=" * 50)

        self.is_running = True

        # 初始化 MCP 客户端
        await self.mcp_client.connect()

        print("\n🎤 语音模式启动，请说话...")
        print("   说 '退出' 或 '结束' 停止\n")

        # 创建事件处理器
        event_handler = VoiceMCPEventHandler(
            on_function_call_done_callback=self._on_function_call_done
        )

        # 创建实时推理器
        tools = self.mcp_client.get_tools_for_realtime()
        self.realtime_inferencer = RealtimeInferencer(
            config=self.realtime_config,
            event_handler=event_handler,
            tools=tools,
        )

        try:
            await self.realtime_inferencer.run(enable_audio_playback=False)
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
