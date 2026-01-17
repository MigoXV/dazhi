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
