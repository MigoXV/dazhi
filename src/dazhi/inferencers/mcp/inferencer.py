"""MCP 推理器模块 - 结合 LLM 和 MCP 工具调用"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from dazhi.inferencers.realtime.inferencer import (
    RealtimeConfig,
    RealtimeEventHandler,
    RealtimeInferencer,
    TranscriptEvent,
    AudioEvent,
    InputTranscriptEvent,
)


@dataclass
class MCPConfig:
    """MCP 配置"""

    mcp_url: str
    mcp_token: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.mcp_token is None:
            self.mcp_token = os.getenv("MCD_MCP_TOKEN", "").strip()
        if self.mcp_token and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.mcp_token}"


@dataclass
class LLMConfig:
    """LLM 配置"""

    base_url: str | None = None
    api_key: str | None = None
    model: str = "qwen3:8B"
    system_prompt: str = "你是一个智能助手，可以帮助用户查询麦当劳相关信息。请用中文回答。"

    def __post_init__(self):
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:10002/v1")
        if self.api_key is None:
            self.api_key = os.getenv("TEST_API_KEY", "dummy_api_key")


class MCPClient:
    """MCP 客户端封装"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.session: ClientSession | None = None
        self.tools: list[types.Tool] = []
        self._read = None
        self._write = None
        self._cm = None

    async def connect(self) -> None:
        """连接到 MCP 服务"""
        self._cm = streamablehttp_client(self.config.mcp_url, headers=self.config.headers)
        self._read, self._write, _ = await self._cm.__aenter__()
        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()

        # 获取可用工具列表
        tools_resp = await self.session.list_tools()
        self.tools = tools_resp.tools
        print(f"✓ MCP connected, available tools: {[t.name for t in self.tools]}")

    async def disconnect(self) -> None:
        """断开 MCP 连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._cm:
            await self._cm.__aexit__(None, None, None)

    def get_tools_for_openai(self) -> list[ChatCompletionToolParam]:
        """将 MCP 工具转换为 OpenAI 格式"""
        openai_tools = []
        for tool in self.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
                },
            })
        return openai_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具"""
        if not self.session:
            raise RuntimeError("MCP session not connected")

        result = await self.session.call_tool(name, arguments=arguments)

        # 提取结果文本
        texts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
            else:
                texts.append(str(block))
        return "\n".join(texts)


class MCPInferencer:
    """MCP 推理器

    结合 LLM 和 MCP 工具调用，实现智能助手。

    Example:
        ```python
        mcp_config = MCPConfig(mcp_url="https://mcp.mcd.cn/mcp-servers/mcd-mcp")
        inferencer = MCPInferencer(mcp_config=mcp_config)
        await inferencer.run()
        ```
    """

    def __init__(
        self,
        mcp_config: MCPConfig,
        llm_config: LLMConfig | None = None,
    ):
        self.mcp_config = mcp_config
        self.llm_config = llm_config or LLMConfig()

        self.mcp_client = MCPClient(mcp_config)
        self.llm_client: AsyncOpenAI | None = None

        self.is_running = False
        self.messages: list[dict[str, Any]] = []

    async def _init_llm_client(self) -> None:
        """初始化 LLM 客户端"""
        self.llm_client = AsyncOpenAI(
            base_url=self.llm_config.base_url,
            api_key=self.llm_config.api_key,
            http_client=httpx.AsyncClient(verify=False),
        )
        print(f"✓ LLM client initialized: {self.llm_config.base_url}")

    async def process_user_input(self, user_input: str) -> str:
        """处理用户输入，调用 LLM 和 MCP 工具"""
        if not self.llm_client:
            return "LLM 客户端未初始化"

        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input+" /no_think"})
        print(f"\n👤 用户: {user_input}")

        # 获取 MCP 工具
        tools = self.mcp_client.get_tools_for_openai()

        # 调用 LLM
        response = await self.llm_client.chat.completions.create(
            model=self.llm_config.model,
            messages=self.messages,
            tools=tools if tools else None,
            stream=True,
        )

        # 处理流式响应
        content_chunks = []
        tool_calls = []

        print("🤖 助手: ", end="", flush=True)
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                content_chunks.append(delta.content)
                print(delta.content, end="", flush=True)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    while len(tool_calls) <= tool_call.index:
                        tool_calls.append({
                            "id": None,
                            "type": "function",
                            "function": {"name": None, "arguments": ""},
                        })
                    target = tool_calls[tool_call.index]
                    if tool_call.id:
                        target["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        target["function"]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        target["function"]["arguments"] += tool_call.function.arguments

        print()  # 换行

        # 构建助手消息
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_chunks),
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.messages.append(assistant_message)

        # 如果有工具调用，执行工具
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}

                print(f"🔧 调用工具: {tool_name}({tool_args})")

                try:
                    result = await self.mcp_client.call_tool(tool_name, tool_args)
                    print(f"📋 工具结果: {result[:200]}..." if len(result) > 200 else f"📋 工具结果: {result}")
                except Exception as e:
                    result = f"工具调用失败: {e}"
                    print(f"❌ {result}")

                # 添加工具结果
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })

            # 再次调用 LLM 获取最终回复
            response = await self.llm_client.chat.completions.create(
                model=self.llm_config.model,
                messages=self.messages,
                stream=True,
            )

            final_content = []
            print("🤖 助手: ", end="", flush=True)
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    final_content.append(delta.content)
                    print(delta.content, end="", flush=True)
            print()

            final_message = {"role": "assistant", "content": "".join(final_content)}
            self.messages.append(final_message)
            return final_message["content"]

        return assistant_message["content"]

    async def run(self) -> None:
        """运行文本模式"""
        print("=" * 50)
        print("🍔 麦当劳 MCP 智能助手")
        print("=" * 50)

        self.is_running = True

        # 初始化
        await self._init_llm_client()
        await self.mcp_client.connect()

        # 初始化消息历史
        self.messages = [
            {"role": "system", "content": self.llm_config.system_prompt}
        ]

        print("\n💬 文本模式启动")
        print("   输入 '退出' 或 'exit' 停止\n")

        try:
            while self.is_running:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("👤 请输入: ")
                    )

                    if user_input.strip().lower() in ["退出", "结束", "exit", "quit"]:
                        print("\n👋 再见!")
                        break

                    if not user_input.strip():
                        continue

                    await self.process_user_input(user_input)

                except EOFError:
                    break

        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，正在停止...")
        finally:
            self.is_running = False
            await self.mcp_client.disconnect()
            print("✓ MCP 连接已关闭")

    async def __aenter__(self) -> "MCPInferencer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.is_running = False
        await self.mcp_client.disconnect()


class VoiceMCPEventHandler(RealtimeEventHandler):
    """语音 MCP 事件处理器 - 将语音转写结果传递给队列"""

    def __init__(self, transcript_queue: asyncio.Queue):
        self.transcript_queue = transcript_queue

    async def on_session_created(self, session_id: str) -> None:
        print(f"✓ Voice session created: {session_id}")

    async def on_session_updated(self) -> None:
        print("✓ Voice session updated")

    async def on_transcript_delta(self, event: TranscriptEvent) -> None:
        """每个 delta 就是一段完整的转写结果"""
        print(f"\n🎤 语音识别: {event.delta}")
        if event.delta.strip():
            await self.transcript_queue.put(event.delta.strip())

    async def on_audio_delta(self, event: AudioEvent) -> None:
        pass  # 不播放音频

    async def on_response_done(self) -> None:
        pass

    async def on_input_transcript_completed(self, event: InputTranscriptEvent) -> None:
        pass  # 不使用这个事件


class VoiceMCPInferencer:
    """语音 + MCP 推理器

    结合实时语音识别和 MCP 工具调用，实现语音控制的智能助手。

    Example:
        ```python
        mcp_config = MCPConfig(mcp_url="https://mcp.mcd.cn/mcp-servers/mcd-mcp")
        inferencer = VoiceMCPInferencer(mcp_config=mcp_config)
        await inferencer.run()
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

        self.mcp_inferencer = MCPInferencer(mcp_config, llm_config)
        self.realtime_inferencer: RealtimeInferencer | None = None

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
                await self.mcp_inferencer.process_user_input(transcript)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"\n❌ 处理转写结果出错: {e}")

    async def run(self) -> None:
        """运行语音 MCP 模式"""
        print("=" * 50)
        print("🍔 麦当劳 MCP 智能助手 (语音模式)")
        print("=" * 50)

        self.is_running = True

        # 初始化 MCP 推理器
        await self.mcp_inferencer._init_llm_client()
        await self.mcp_inferencer.mcp_client.connect()
        self.mcp_inferencer.messages = [
            {"role": "system", "content": self.mcp_inferencer.llm_config.system_prompt}
        ]
        self.mcp_inferencer.is_running = True

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
            await self.mcp_inferencer.mcp_client.disconnect()
            print("✓ 所有连接已关闭")

    async def __aenter__(self) -> "VoiceMCPInferencer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.is_running = False
        if self.realtime_inferencer:
            await self.realtime_inferencer.stop()
        await self.mcp_inferencer.mcp_client.disconnect()
