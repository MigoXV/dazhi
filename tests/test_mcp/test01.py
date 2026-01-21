import asyncio
import os

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client


MCD_MCP_URL = "https://mcp.mcd.cn/mcp-servers/mcd-mcp"


def _print_tool_result(result: types.CallToolResult) -> None:
    # MCP tool result 通常是 content blocks（text / image / etc）
    for block in result.content:
        # 兼容处理：不同版本 SDK block 类型可能略有差异
        text = getattr(block, "text", None)
        if isinstance(text, str):
            print(text)
        else:
            print(block)


async def main() -> None:
    token = os.getenv("MCD_MCP_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请先设置环境变量 MCD_MCP_TOKEN")

    headers = {"Authorization": f"Bearer {token}"}

    # 连接远程 MCP（Streamable HTTP），并建立 session
    async with streamablehttp_client(MCD_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1) 列出可用工具
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            print("Available tools:", tool_names)

            # 2) 调用一个“无副作用”的工具：now-time-info（返回当前时间信息）
            # 工具名/描述来自官方工具列表：now-time-info / campaign-calender / available-coupons 等 :contentReference[oaicite:3]{index=3}
            print("\n--- call now-time-info ---")
            r1 = await session.call_tool("now-time-info", arguments={})
            _print_tool_result(r1)

            # 3) 再示例一个：活动日历 campaign-calender（不传参=默认查当月；也可传 specifiedDate=yyyy-MM-dd） :contentReference[oaicite:4]{index=4}
            print("\n--- call campaign-calender (specifiedDate=2026-01-15) ---")
            r2 = await session.call_tool(
                "campaign-calender",
                arguments={"specifiedDate": "2026-01-15"},
            )
            _print_tool_result(r2)


if __name__ == "__main__":
    asyncio.run(main())
