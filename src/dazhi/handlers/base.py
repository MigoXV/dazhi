from abc import ABC, abstractmethod
from typing import Any


class RealtimeEventHandler(ABC):
    """实时事件处理器抽象基类"""

    @abstractmethod
    async def handle_event(self, event: Any, connection: Any = None) -> None:
        """处理实时事件的抽象方法"""
        pass
