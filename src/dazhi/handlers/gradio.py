import logging
from typing import Any, Dict, List, Optional

from dazhi.codec import AudioPlayerAsync
from dazhi.message.manager import MessageManager

from .base import RealtimeEventHandler

logger = logging.getLogger(__name__)


class GradioEventHandler(RealtimeEventHandler):
    """默认事件处理器 - 打印到控制台"""

    def __init__(
        self,
        audio_player: AudioPlayerAsync | None = None,
        tool_executors: Optional[Dict[str, Any]] = None,
    ):
        self.audio_player = audio_player
        self.message_manager = MessageManager(tool_executors=tool_executors)
        self.chatbot_history: List[Dict[str, str]] = []

    async def handle_event(self, event: Any, connection: Any = None) -> None:
        await self.message_manager.handle_event(event, connection)

    def get_history(self) -> List[Dict[str, str]]:
        """获取聊天历史记录"""
        return self.message_manager.render_gradio_history()
