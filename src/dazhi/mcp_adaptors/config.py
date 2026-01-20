import os
from dataclasses import dataclass, field


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
