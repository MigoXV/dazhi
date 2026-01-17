"""LLM 推理器模块 - 仅负责 LLM 对话推理"""

import asyncio
from typing import Any

import httpx
from openai import AsyncOpenAI

from dazhi.inferencers.llm.session import LLMChatSession
from dazhi.mcp_adaptors.config import LLMConfig


class LLMInferencer:
    """LLM 推理器

    仅负责 LLM 对话推理，不涉及 MCP 工具调用。

    Example:
        ```python
        llm_config = LLMConfig()
        inferencer = LLMInferencer(llm_config=llm_config)
        await inferencer.run()
        ```
    """

    def __init__(
        self,
        llm_config: LLMConfig,
    ):
        self.llm_config = llm_config
        self.llm_client: AsyncOpenAI | None = None
        self.is_running = False

    async def _init_llm_client(self) -> None:
        """初始化 LLM 客户端"""
        self.llm_client = AsyncOpenAI(
            base_url=self.llm_config.base_url,
            api_key=self.llm_config.api_key,
            http_client=httpx.AsyncClient(verify=False),
        )
        print(f"✓ LLM client initialized: {self.llm_config.base_url}")

    async def process_user_input(
        self,
        session: LLMChatSession,
        user_input: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """处理用户输入，仅调用 LLM 推理"""
        if not self.llm_client:
            return {"role": "assistant", "content": "LLM 客户端未初始化"}

        # 添加用户消息
        session.add_user(user_input)
        print(f"\n👤 用户: {user_input}")

        assistant_message = await self._stream_completion(session=session, tools=tools)
        session.add_assistant(assistant_message)
        return assistant_message

    async def continue_assistant(self, session: LLMChatSession) -> dict[str, Any]:
        """继续基于当前上下文生成回复"""
        if not self.llm_client:
            return {"role": "assistant", "content": "LLM 客户端未初始化"}

        assistant_message = await self._stream_completion(session=session)
        session.add_assistant(assistant_message)
        return assistant_message

    async def _stream_completion(
        self,
        session: LLMChatSession,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        response = await self.llm_client.chat.completions.create(
            model=self.llm_config.model,
            messages=session.messages,
            tools=tools if tools else None,
            stream=True,
        )

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
                        tool_calls.append(
                            {
                                "id": None,
                                "type": "function",
                                "function": {"name": None, "arguments": ""},
                            }
                        )
                    target = tool_calls[tool_call.index]
                    if tool_call.id:
                        target["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        target["function"]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        target["function"]["arguments"] += tool_call.function.arguments

        print()

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_chunks),
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    async def run(self) -> None:
        """运行文本模式"""
        print("=" * 50)
        print("🍔 LLM 智能助手")
        print("=" * 50)

        self.is_running = True

        # 初始化
        await self._init_llm_client()

        # 初始化会话
        session = LLMChatSession(system_prompt=self.llm_config.system_prompt)

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

                    await self.process_user_input(session, user_input)

                except EOFError:
                    break

        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，正在停止...")
        finally:
            self.is_running = False

    async def __aenter__(self) -> "LLMInferencer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.is_running = False
