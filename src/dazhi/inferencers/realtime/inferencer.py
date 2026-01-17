"""实时推理器模块"""

import asyncio
import os
import ssl
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, List, Optional

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (
    ResponseOutputItemAddedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    RealtimeConversationItemFunctionCallOutput,
)

from dazhi.codec.audio import (
    CHANNELS,
    SAMPLE_RATE,
    AudioPlayerAsync,
    AudioRecorder,
    decode_audio_from_base64,
    encode_audio_to_base64,
    query_audio_devices,
)
from dazhi.handlers import DefaultEventHandler, RealtimeEventHandler
from openai.types.realtime import RealtimeFunctionTool


@dataclass
class RealtimeConfig:
    """实时推理配置"""

    base_url: str | None = None
    api_key: str | None = None
    model: str = "transcribe"
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    ssl_verify: bool = False
    channels: int = CHANNELS
    sample_rate: int = SAMPLE_RATE
    read_interval: float = 0.02  # 20ms

    def __post_init__(self):
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


class RealtimeInferencer:

    def __init__(
        self,
        config: RealtimeConfig | None = None,
        event_handler: RealtimeEventHandler | None = None,
        tools: Optional[List[RealtimeFunctionTool]] = None,
    ):
        """
        初始化实时推理器

        Args:
            config: 推理配置，如果为None则使用默认配置
            event_handler: 事件处理器，如果为None则使用默认处理器
            tools: 可选的实时函数工具列表
        """
        self.config = config or RealtimeConfig()
        self.tools = tools or []

        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self.audio_player: AudioPlayerAsync | None = None
        self.event_handler = event_handler
        self.connection: AsyncRealtimeConnection | None = None
        self.is_running = False
        self._accumulated_items: dict[str, str] = {}

    def _create_ssl_context(self) -> ssl.SSLContext:
        """创建 SSL 上下文"""
        ssl_context = ssl.create_default_context()
        if not self.config.ssl_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context

    def _get_event_handler(self) -> RealtimeEventHandler:
        """获取事件处理器"""
        if not self.event_handler:
            self.event_handler = DefaultEventHandler(self.audio_player)
        return self.event_handler

    async def _handle_event(self, event: Any) -> None:
        """处理单个事件"""
        handler = self._get_event_handler()
        # 创建会话事件
        if isinstance(event, SessionCreatedEvent):
            await handler.on_session_created(event.session.id)
            return
        # 会话更新事件
        if isinstance(event, SessionUpdatedEvent):
            await handler.on_session_updated()
            return
        # 处理对话创建事件
        if isinstance(event, ResponseOutputItemAddedEvent):
            await handler.on_response_output_item_add(event)
            return
        # 处理 function call 参数增量事件（打字机效果）
        if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            await handler.on_function_call_delta(event)
            return
        # 处理 function call 参数完成事件
        if isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            result = await handler.on_function_call_done(event)
            if result is not None:
                await self.connection.conversation.item.create(
                    item=RealtimeConversationItemFunctionCallOutput(
                        call_id=event.call_id,
                        output=result,
                        type="function_call_output",
                    )
                )
            return
        # 处理音频转文本增量事件
        if isinstance(event, ResponseAudioTranscriptDeltaEvent):
            await handler.on_transcript_delta(event)
            return
        # 处理用户语音转写完成事件
        if isinstance(event, ConversationItemInputAudioTranscriptionCompletedEvent):
            await handler.on_input_audio_transcription_completed(event)
            return
        # 处理 LLM 文本响应增量事件（打字机效果）
        if isinstance(event, ResponseTextDeltaEvent):
            await handler.on_text_delta(event)
            return
        # 处理 LLM 文本响应完成事件（对话结束标记）
        if isinstance(event, ResponseTextDoneEvent):
            await handler.on_text_done(event)
            return
        print(f"Unhandled event type: {type(event)}")
        return

    async def _handle_connection(self) -> None:
        """处理实时连接"""
        ssl_context = self._create_ssl_context()

        async with self.client.realtime.connect(
            model=self.config.model,
            websocket_connection_options={"ssl": ssl_context},
        ) as conn:
            self.connection = conn
            print("✓ Connected to Realtime API")

            # 配置会话
            await conn.session.update(
                session=RealtimeSessionCreateRequest(
                    type="realtime",
                    output_modalities=self.config.output_modalities,
                    tools=self.tools,
                    model=self.config.model,
                )
            )
            print("✓ Session configured")

            async for event in conn:
                if not self.is_running:
                    break
                await self._handle_event(event)

    async def _send_audio(self) -> None:
        """发送麦克风音频"""
        sent_audio = False
        read_size = int(self.config.sample_rate * self.config.read_interval)

        device_info = query_audio_devices()
        print(f"Audio devices: {device_info}")

        recorder = AudioRecorder(
            channels=self.config.channels,
            sample_rate=self.config.sample_rate,
        )
        recorder.start()
        print("✓ Audio stream started - now recording...")

        try:
            while self.is_running:
                if recorder.read_available() < read_size:
                    await asyncio.sleep(0)
                    continue

                data, _ = recorder.read(read_size)

                if self.connection is None:
                    await asyncio.sleep(0.1)
                    continue

                if not sent_audio:
                    asyncio.create_task(
                        self.connection.send({"type": "response.cancel"})
                    )
                    sent_audio = True

                await self.connection.input_audio_buffer.append(
                    audio=encode_audio_to_base64(data)
                )

                await asyncio.sleep(0)
        finally:
            recorder.stop()
            recorder.close()
            print("✓ Audio stream closed")

    async def run(self, enable_audio_playback: bool = True) -> None:
        """
        运行推理器

        Args:
            enable_audio_playback: 是否启用音频播放
        """
        print("=== Realtime Inferencer ===")
        print("Starting audio streaming... Press Ctrl+C to stop")
        print()

        self.is_running = True
        self._accumulated_items.clear()

        if enable_audio_playback:
            self.audio_player = AudioPlayerAsync()

        try:
            await asyncio.gather(
                self._handle_connection(),
                self._send_audio(),
            )
        except KeyboardInterrupt:
            print("\n✓ Stopping...")
        finally:
            self.is_running = False
            if self.audio_player:
                self.audio_player.close()
                print("✓ Audio player closed")

    async def stop(self) -> None:
        """停止推理器"""
        self.is_running = False

    async def __aenter__(self) -> "RealtimeInferencer":
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器退出"""
        await self.stop()
        if self.audio_player:
            self.audio_player.close()
