from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

from openai.types.realtime import (
    ResponseOutputItemAddedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    RealtimeConversationItemFunctionCallOutput,
)

# 回调类型定义：(function_name, event) -> result
FunctionCallDoneCallback = Callable[
    [str, ResponseFunctionCallArgumentsDoneEvent],
    Coroutine[Any, Any, str | None],
]


class RealtimeEventHandler(ABC):
    """实时事件处理器抽象基类"""

    @abstractmethod
    async def on_session_created(self, session_id: str) -> None:
        """会话创建时调用"""
        pass

    @abstractmethod
    async def on_session_updated(self) -> None:
        """会话更新时调用"""
        pass

    @abstractmethod
    async def on_response_output_item_add(self, event: ResponseOutputItemAddedEvent) -> None:
        """对话创建时调用"""
        pass

    @abstractmethod
    async def on_function_call_delta(self, event: ResponseFunctionCallArgumentsDeltaEvent) -> None:
        """收到 function call 参数增量时调用（打字机效果）"""
        pass

    @abstractmethod
    async def on_function_call_done(
        self, event: ResponseFunctionCallArgumentsDoneEvent
    ) -> str | None:
        """function call 参数输出完成时调用"""
        pass

    @abstractmethod
    async def on_transcript_delta(self, event) -> None:
        """收到音频转文本增量时调用（ResponseAudioTranscriptDeltaEvent）"""
        pass

    @abstractmethod
    async def on_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        """收到 LLM 文本响应增量时调用（打字机效果）"""
        pass

    @abstractmethod
    async def on_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        """收到音频增量时调用"""
        pass

    @abstractmethod
    async def on_response_done(self) -> None:
        """响应完成时调用"""
        pass

    @abstractmethod
    async def on_input_audio_transcription_completed(
        self,
        event: ConversationItemInputAudioTranscriptionCompletedEvent,
    ) -> None:
        """输入音频转录完成时调用（用户语音转写结果）"""
        pass

    @abstractmethod
    async def on_text_done(
        self,
        event: ResponseTextDoneEvent,
    ) -> None:
        """文本响应完成时调用（服务端 LLM 对话结果）"""
        pass

    async def handle_event(self, event: Any, connection: Any = None) -> None:
        """处理单个事件"""
        # 创建会话事件
        if isinstance(event, SessionCreatedEvent):
            await self.on_session_created(event.session.id)
            return
        # 会话更新事件
        if isinstance(event, SessionUpdatedEvent):
            await self.on_session_updated()
            return
        # 处理对话创建事件
        if isinstance(event, ResponseOutputItemAddedEvent):
            if event.item.type == "tool":
                await self.on_response_output_item_add(event)
                return
        # 处理 function call 参数增量事件（打字机效果）
        if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            await self.on_function_call_delta(event)
            return
        # 处理 function call 参数完成事件
        if isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            result = await self.on_function_call_done(event)
            if result is not None and connection is not None:
                await connection.conversation.item.create(
                    item=RealtimeConversationItemFunctionCallOutput(
                        call_id=event.call_id,
                        output=result,
                        type="function_call_output",
                    )
                )
            return
        # 处理音频转文本增量事件
        if isinstance(event, ResponseAudioTranscriptDeltaEvent):
            await self.on_transcript_delta(event)
            return
        # 处理用户语音转写完成事件
        if isinstance(event, ConversationItemInputAudioTranscriptionCompletedEvent):
            await self.on_input_audio_transcription_completed(event)
            return
        # 处理 LLM 文本响应增量事件（打字机效果）
        if isinstance(event, ResponseTextDeltaEvent):
            await self.on_text_delta(event)
            return
        # 处理 LLM 文本响应完成事件（对话结束标记）
        if isinstance(event, ResponseTextDoneEvent):
            await self.on_text_done(event)
            return
        # print(f"{event}")
        return
