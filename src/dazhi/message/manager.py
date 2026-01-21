import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from .history_item import (
    AssistantMessageHistoryItem,
    MessageHistoryItem,
    SystemMessageHistoryItem,
    UserMessageHistoryItem,
)

logger = logging.getLogger(__name__)


class MessageManager:

    def __init__(self):
        self.message_histories: OrderedDict[str, MessageHistoryItem] = OrderedDict()

    def handle_event(self, event: Any, connection: Any = None) -> None:
        event_type: str = event.type
        names = event_type.split(".")
        if names[0] == "session":
            # 建立连接
            message_id = "session_init"
            system_item = self.update_message(message_id, "system")
            if names[1] == "created":
                system_item.connect()
            elif names[1] == "updated":
                system_item.update(event.session)
            else:
                logger.info(f"Unhandled session event: {event}")
        elif names[0] == "input_audio_buffer":  # VAD 事件
            item_id = event.item_id
            message_item = self.update_message(item_id, "user")
            self.route_vad(names, message_item, event)
        elif names[0] == "response":
            if names[1] == "output_text":
                message_item = self.update_message(event.item_id, "assistant")
                self.route_response_text(names, message_item, event)
            else:
                logger.info(f"Unhandled response event: {event_type}")
        elif names[0] == "conversation":
            self.route_conversation(names, event)
        elif names[0] == "rate_limits":
            logger.info(event_type)
        else:
            logger.info(f"Unhandled event type: {event_type}")
        return

    def render_gradio_history(self) -> List[Dict[str, str]]:
        """渲染消息历史以适应 Gradio 聊天界面"""
        rendered_history = []
        for item in self.message_histories.values():
            content = item.render()
            if content:
                rendered_history.append({"role": item.role, "content": content})
        return rendered_history

    def update_message(
        self,
        message_id: str,
        role: Optional[Literal["system", "user", "assistant"]] = None,
    ) -> MessageHistoryItem:
        """如消息存在，找到对应消息并返回，否则创建新消息"""
        if message_id in self.message_histories:
            return self.message_histories[message_id]
        if role == "system":
            item = SystemMessageHistoryItem()
        elif role == "user":
            item = UserMessageHistoryItem()
        elif role == "assistant":
            item = AssistantMessageHistoryItem()
        else:
            raise ValueError(f"Unknown role: {role}")
        self.message_histories[message_id] = item
        return item

    def route_vad(self, names: List[str], message_item: UserMessageHistoryItem, event):
        if names[1] == "speech_started":
            message_item.start_time = event.audio_start_ms / 1000
        elif names[1] == "speech_stopped":
            message_item.end_time = event.audio_end_ms / 1000
        else:
            logger.info(f"Unhandled input_audio_buffer event: {event}")

    def route_conversation(self, names: List[str], event):
        if names[1] == "item":
            if names[2] == "input_audio_transcription":
                item_id = event.item_id
                message_item = self.update_message(item_id)
                if names[3] == "delta":
                    message_item.update_stream(event.delta)
                elif names[3] == "completed":
                    message_item.finish()
                else:
                    logger.info(
                        f"Unhandled conversation.input_audio_transcription event: {event}"
                    )
        else:
            logger.info(f"Unhandled conversation event: {event}")

    def route_response_text(
        self, names: List[str], message_item: UserMessageHistoryItem, event
    ):
        if names[2] == "delta":
            message_item.update_stream(event.delta)
        elif names[2] == "done":
            message_item.finish()
        else:
            logger.info(f"Unhandled response.output_text event: {event}")
