"""实时推理器模块"""

import asyncio
import os
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime.realtime_session_create_request import \
    RealtimeSessionCreateRequest

from dazhi.codec.audio import (CHANNELS, SAMPLE_RATE, AudioPlayerAsync,
                               AudioRecorder, decode_audio_from_base64,
                               encode_audio_to_base64, query_audio_devices)


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


@dataclass
class TranscriptEvent:
    """转录事件"""

    item_id: str
    delta: str
    accumulated_text: str


@dataclass
class AudioEvent:
    """音频事件"""

    item_id: str
    audio_data: bytes


@dataclass
class InputTranscriptEvent:
    """输入音频转写完成事件"""

    item_id: str
    transcript: str


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
    async def on_transcript_delta(self, event: TranscriptEvent) -> None:
        """收到转录增量时调用"""
        pass

    @abstractmethod
    async def on_audio_delta(self, event: AudioEvent) -> None:
        """收到音频增量时调用"""
        pass

    @abstractmethod
    async def on_response_done(self) -> None:
        """响应完成时调用"""
        pass

    @abstractmethod
    async def on_input_transcript_completed(self, event: InputTranscriptEvent) -> None:
        """输入音频转写完成时调用 - 用于语音识别结果"""
        pass


class DefaultEventHandler(RealtimeEventHandler):
    """默认事件处理器 - 打印到控制台"""

    def __init__(self, audio_player: AudioPlayerAsync | None = None):
        self.audio_player = audio_player
        self.last_audio_item_id: str | None = None

    async def on_session_created(self, session_id: str) -> None:
        print(f"✓ Session created: {session_id}")

    async def on_session_updated(self) -> None:
        print("✓ Session updated")

    async def on_transcript_delta(self, event: TranscriptEvent) -> None:
        print(f"\rTranscript: {event.delta}")

    async def on_audio_delta(self, event: AudioEvent) -> None:
        if self.audio_player:
            if event.item_id != self.last_audio_item_id:
                self.audio_player.reset_frame_count()
                self.last_audio_item_id = event.item_id
            self.audio_player.add_data(event.audio_data)

    async def on_response_done(self) -> None:
        print()  # New line after transcript

    async def on_input_transcript_completed(self, event: InputTranscriptEvent) -> None:
        print(f"\n✓ 语音识别完成: {event.transcript}")


class RealtimeInferencer:
    """实时推理器

    用于处理实时音频流并获取转录/响应结果。

    Example:
        ```python
        config = RealtimeConfig(model="transcribe")
        inferencer = RealtimeInferencer(config)
        await inferencer.run()
        ```
    """

    def __init__(
        self,
        config: RealtimeConfig | None = None,
        event_handler: RealtimeEventHandler | None = None,
    ):
        """
        初始化实时推理器

        Args:
            config: 推理配置，如果为None则使用默认配置
            event_handler: 事件处理器，如果为None则使用默认处理器
        """
        self.config = config or RealtimeConfig()
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
        if self.event_handler:
            return self.event_handler
        return DefaultEventHandler(self.audio_player)

    async def _handle_event(self, event: Any) -> None:
        """处理单个事件"""
        handler = self._get_event_handler()

        if event.type == "session.created":
            await handler.on_session_created(event.session.id)
            return

        if event.type == "session.updated":
            await handler.on_session_updated()
            return

        if event.type == "response.output_audio.delta":
            audio_data = decode_audio_from_base64(event.delta)
            await handler.on_audio_delta(
                AudioEvent(item_id=event.item_id, audio_data=audio_data)
            )
            return

        if event.type == "response.output_audio_transcript.delta":
            # 累积文本
            if event.item_id not in self._accumulated_items:
                self._accumulated_items[event.item_id] = event.delta
            else:
                self._accumulated_items[event.item_id] += event.delta

            await handler.on_transcript_delta(
                TranscriptEvent(
                    item_id=event.item_id,
                    delta=event.delta,
                    accumulated_text=self._accumulated_items[event.item_id],
                )
            )
            return

        if event.type == "response.done":
            await handler.on_response_done()
            return

        # 处理输入音频转写完成事件 (语音识别结果)
        if event.type == "conversation.item.input_audio_transcription.completed":
            await handler.on_input_transcript_completed(
                InputTranscriptEvent(
                    item_id=event.item_id,
                    transcript=event.transcript,
                )
            )
            return

        # 打印未处理的事件类型（调试用）
        # print(f"[DEBUG] Unhandled event: {event.type}")

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


async def main():
    """示例入口"""
    config = RealtimeConfig()
    inferencer = RealtimeInferencer(config)
    await inferencer.run()


if __name__ == "__main__":
    asyncio.run(main())
