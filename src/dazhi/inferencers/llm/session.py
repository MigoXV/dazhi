"""LLM 会话模块 - 管理消息历史"""

from typing import Any


class LLMChatSession:
    """LLM 会话

    负责管理对话消息历史，支持追加不同角色消息与重置。
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.messages: list[dict[str, Any]] = list(messages) if messages else []
        if not self.messages and system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_user(self, content: str, *, no_think: bool = True) -> dict[str, Any]:
        if no_think:
            content = f"{content} /no_think"
        message = {"role": "user", "content": content}
        self.messages.append(message)
        return message

    def add_assistant(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> dict[str, Any]:
        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.messages.append(message)
        return message

    def reset(self, system_prompt: str | None = None) -> None:
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
