from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class MessageHistoryItem(ABC):
    role: str
    _content: Optional[str] = field(default=None, repr=False)
    pending: bool = field(default=True, repr=False)

    def render_cached(self) -> str:
        if not self.pending:
            return self._content or ""
        return ""

    def render(self) -> str:
        content = self.render_cached()
        if content:
            return content
        content = self.render_interm()
        return content

    def finish(self):
        self.pending = False
        self._content = self.render()


@dataclass
class SystemMessageHistoryItem(MessageHistoryItem):
    role: Literal["system"] = field(
        default="assistant", init=False
    )  # 为了兼容 Gradio 聊天界面，系统消息也设为 assistant 角色
    connected: bool = False
    config: Optional[Any] = None

    def render(self) -> str:
        content = ""
        if self.connected:
            content += "[System connected]"
        if self.config:
            content += f"\nConfig: {str(self.config)}"
        return content

    def connect(self):
        self.connected = True

    def update(self, config: Any):
        self.connected = True
        self.config = config


@dataclass
class UserMessageHistoryItem(MessageHistoryItem):
    role: Literal["user"] = field(default="user", init=False)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    transcription: Optional[str] = None

    def render_interm(self) -> str:
        head = self.render_head()
        if head:
            return f"{head}\n{self.transcription}"
        return self.transcription or ""

    def render_head(self) -> str:
        start_time = f"{self.start_time:.2f}" if self.start_time is not None else ""
        end_time = f"{self.end_time:.2f}" if self.end_time is not None else ""
        return f"[{start_time} - {end_time}] " if start_time or end_time else ""

    def update_stream(self, transcription_delta: str) -> None:
        self.transcription = (self.transcription or "") + transcription_delta


@dataclass
class AssistantMessageHistoryItem(MessageHistoryItem):
    role: Literal["assistant"] = field(default="assistant", init=False)
    chat_content: Optional[str] = None

    def render_interm(self) -> str:
        return self.chat_content or ""

    def update_stream(self, chat_content_delta: str) -> None:
        self.chat_content = (self.chat_content or "") + chat_content_delta
