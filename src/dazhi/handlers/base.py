from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

from openai.types.realtime import (
    ResponseOutputItemAddedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseAudioDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

# 回调类型定义：(function_name, event) -> result
FunctionCallDoneCallback = Callable[
    [str, ResponseFunctionCallArgumentsDoneEvent], 
    Coroutine[Any, Any, str | None]
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
    async def on_function_call_done(self, event: ResponseFunctionCallArgumentsDoneEvent) -> None:
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
