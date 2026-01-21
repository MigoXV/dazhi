"""MCP 客户端模块 - 适配 Realtime 工具调用"""

from typing import Any

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from openai.types.realtime import RealtimeFunctionTool

from .config import MCPConfig


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
        self._cm = streamablehttp_client(
            self.config.mcp_url, headers=self.config.headers
        )
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

    def get_tools_for_realtime(self) -> list[RealtimeFunctionTool]:
        """将 MCP 工具转换为 Realtime 格式"""
        realtime_tools: list[RealtimeFunctionTool] = []
        for tool in self.tools:
            realtime_tools.append(
                RealtimeFunctionTool(
                    type="function",
                    name=tool.name,
                    description=tool.description or "",
                    parameters=(
                        tool.inputSchema
                        if tool.inputSchema
                        else {"type": "object", "properties": {}, "required": []}
                    ),
                )
            )
        return realtime_tools

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
