import asyncio

from openai.types.realtime import (
    ResponseOutputItemAddedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseAudioDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from dazhi.handlers import RealtimeEventHandler
from dazhi.handlers.base import FunctionCallDoneCallback


class VoiceMCPEventHandler(RealtimeEventHandler):
    """语音 MCP 事件处理器 - 将语音转写结果传递给队列"""

    def __init__(
        self,
        transcript_queue: asyncio.Queue | None = None,
        on_function_call_done_callback: FunctionCallDoneCallback | None = None,
    ):
        self.transcript_queue = transcript_queue
        self._on_function_call_done_callback = on_function_call_done_callback
        self._current_function_name: str | None = None  # 当前调用的函数名

    async def on_session_created(self, session_id: str) -> None:
        print(f"✓ Voice session created: {session_id}")

    async def on_session_updated(self) -> None:
        print("✓ Voice session updated")

    async def on_response_output_item_add(self, event: ResponseOutputItemAddedEvent) -> None:
        """对话创建时调用 - 从这里提取函数名"""
        # 从 item 中提取函数名
        if hasattr(event, "item") and hasattr(event.item, "name"):
            self._current_function_name = event.item.name

    async def on_function_call_delta(self, event: ResponseFunctionCallArgumentsDeltaEvent) -> None:
        """function call 参数增量 - 打字机效果"""
        print(event.delta, end="", flush=True)

    async def on_function_call_done(self, event: ResponseFunctionCallArgumentsDoneEvent) -> str | None:
        """function call 参数输出完成时调用"""
        print()  # 换行
        # 调用回调函数（如果有的话），传递函数名
        if self._on_function_call_done_callback and self._current_function_name:
            result = await self._on_function_call_done_callback(self._current_function_name, event)
            if result:
                print(f"   📋 结果: {result[:200]}..." if len(result) > 200 else f"   📋 结果: {result}")
            self._current_function_name = None
            return result
        
        # 重置函数名
        self._current_function_name = None
        return None

    async def on_transcript_delta(self, event) -> None:
        """音频转文本增量（ResponseAudioTranscriptDeltaEvent）"""
        print(f"\r📝 转写: {event.delta}", end="", flush=True)

    async def on_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        """LLM 文本响应增量 - 打字机效果"""
        print(event.delta, end="", flush=True)

    async def on_audio_delta(self, event) -> None:
        pass  # 不播放音频

    async def on_response_done(self) -> None:
        pass

    async def on_input_transcript_completed(self, event) -> None:
        pass  # 不使用这个事件

    async def on_input_audio_transcription_completed(
        self,
        event: ConversationItemInputAudioTranscriptionCompletedEvent,
    ) -> None:
        """输入音频转录完成时调用（用户语音转写结果）"""
        print(f"\n🎤 语音识别: {event.transcript}")
        if self.transcript_queue is not None and event.transcript.strip():
            await self.transcript_queue.put(event.transcript.strip())

    async def on_text_done(
        self,
        event: ResponseTextDoneEvent,
    ) -> None:
        """文本响应完成时调用（对话结束标记）"""
        print()  # 换行
        # 服务端已完成 LLM 对话，这里只作为对话结束标记
