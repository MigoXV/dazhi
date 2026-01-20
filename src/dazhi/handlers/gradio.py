from typing import Any, Dict, List, Optional

from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    RealtimeConversationItemFunctionCallOutput,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)

from dazhi.codec import AudioPlayerAsync

from .base import FunctionCallDoneCallback, RealtimeEventHandler


class GradioEventHandler(RealtimeEventHandler):
    """默认事件处理器 - 打印到控制台"""

    def __init__(
        self,
        chatbot_history: List[Dict[str, str]],
        audio_player: AudioPlayerAsync | None = None,
        on_function_call_done_callback: FunctionCallDoneCallback | None = None,
    ):
        self.audio_player = audio_player
        self.last_audio_item_id: str | None = None
        self._text_started = False  # 标记是否已开始文本输出
        self._function_call_started = False  # 标记是否已开始 function call 输出
        self._current_function_name: str | None = None  # 当前调用的函数名
        self._on_function_call_done_callback = on_function_call_done_callback

        self.chatbot_history = chatbot_history    
        
    async def on_session_created(self, session_id: str) -> None:
        # print(f"✓ Session created: {session_id}")
        pass

    async def on_session_updated(self) -> None:
        # print("✓ Session updated")
        pass

    async def on_response_output_item_add(
        self, event: ResponseOutputItemAddedEvent
    ) -> None:
        """对话创建时调用 - 从这里提取函数名"""
        # 从 conversation.item 中提取函数名
        self._current_function_name = event.item.name
        # print(f"calling function: {self._current_function_name}")

    async def on_function_call_delta(
        self, event: ResponseFunctionCallArgumentsDeltaEvent
    ) -> None:
        """function call 参数增量 - 打字机效果"""
        if not self._function_call_started:
            func_name = self._current_function_name or "unknown"
            # print(f"\n🔧 调用工具: {func_name} (call_id: {event.call_id})", flush=True)
            # print("   参数: ", end="", flush=True)
            self._function_call_started = True
        # print(event.delta, end="", flush=True)

    async def on_function_call_done(
        self, event: ResponseFunctionCallArgumentsDoneEvent
    ) -> str | None:
        """function call 参数输出完成时调用"""
        if self._function_call_started:
            # print()  # 换行
            self._function_call_started = False

        # 调用回调函数（如果有的话），传递函数名
        if self._on_function_call_done_callback and self._current_function_name:
            result = await self._on_function_call_done_callback(
                self._current_function_name, event
            )
            self._current_function_name = None
            return result

        # 重置函数名
        self._current_function_name = None
        return None

    async def on_transcript_delta(self, event) -> None:
        """音频转文本增量（ResponseAudioTranscriptDeltaEvent）"""
        # print(f"\r📝 转写: {event.delta}", end="", flush=True)

    async def on_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        """LLM 文本响应增量 - 打字机效果"""
        if not self._text_started:
            # print("\n🤖 助手: ", end="", flush=True)
            self._text_started = True
        # print(event.delta, end="", flush=True)

    async def on_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        if self.audio_player:
            if event.item_id != self.last_audio_item_id:
                self.audio_player.reset_frame_count()
                self.last_audio_item_id = event.item_id
            self.audio_player.add_data(event.audio_data)

    async def on_response_done(self) -> None:
        pass
        # print()  # New line after transcript

    async def on_input_audio_transcription_completed(
        self,
        event: ConversationItemInputAudioTranscriptionCompletedEvent,
    ) -> None:
        """输入音频转录完成时调用（用户语音转写结果）"""
        pass
        # print(f"\n🎤 语音识别: {event.transcript}")

    async def on_text_done(
        self,
        event: ResponseTextDoneEvent,
    ) -> None:
        """文本响应完成时调用（对话结束标记）"""
        if self._text_started:
            # print()  # 换行
            pass
            self._text_started = False

    async def handle_event(self, event: Any, connection: Any = None) -> None:
        """处理单个事件"""
        print(event)
        self.chatbot_history.append(
            {"role": "assistant", "content": str(event)}
        )
        return
