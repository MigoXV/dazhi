"""实时推理器模块"""

import asyncio
import logging
import ssl
from typing import Any, Callable, List, Optional

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import RealtimeFunctionTool, RealtimeSessionCreateRequest

from dazhi.codec import AudioPlayerAsync, AudioRecorder, encode_audio_to_base64
from dazhi.handlers import RealtimeEventHandler

from .config import RealtimeConfig

logger = logging.getLogger(__name__)


class RealtimeInferencer:

    def __init__(
        self,
        config: RealtimeConfig,
        event_handler: RealtimeEventHandler,
        audio_recorder: AudioRecorder,
        tools: List[RealtimeFunctionTool] = [],
    ):
        """
        初始化实时推理器

        Args:
            config: 推理配置，如果为None则使用默认配置
            event_handler: 事件处理器，如果为None则使用默认处理器
            tools: 可选的实时函数工具列表
        """
        self.config = config
        self.read_size = int(
            self.config.audio.sample_rate * self.config.audio.read_interval
        )

        self.event_handler = event_handler
        self.tools = tools
        self.client = AsyncOpenAI(
            base_url=self.config.openai.base_url,
            api_key=self.config.openai.api_key,
        )
        self.audio_player = audio_recorder

        self.connection: AsyncRealtimeConnection | None = None

        self.is_running = False
        self._connected = asyncio.Event()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """创建 SSL 上下文"""
        ssl_context = ssl.create_default_context()
        if not self.config.connection.ssl_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context

    async def connect(self) -> None:
        """处理实时连接"""
        try:
            ssl_context = self._create_ssl_context()
            async with self.client.realtime.connect(
                model=self.config.connection.model,
                websocket_connection_options={"ssl": ssl_context},
            ) as conn:
                self.connection = conn
                logger.info("✓ Connected to Realtime API")

                # 配置会话
                await conn.session.update(
                    session=RealtimeSessionCreateRequest(
                        type="realtime",
                        output_modalities=self.config.session.output_modalities,
                        tools=self.tools,
                        model=self.config.connection.model,
                        audio=self.config.session.audio,
                    )
                )
                logger.info("✓ Session configured")

                self._connected.set()
                async for event in conn:
                    if not self.is_running:
                        break
                    await self.event_handler.handle_event(event, self.connection)
        except Exception as e:
            logger.error(f"Error in RealtimeInferencer connection: {e}")
        finally:
            self.connection = None
            self._connected.clear()
            logger.info("✓ Disconnected from Realtime API")

    async def send_audio_loop(self, enable_audio_playback: bool = True) -> None:
        """循环发送音频数据"""
        logger.info(
            f"send_audio_loop started, enable_audio_playback={enable_audio_playback}"
        )
        self.is_running = True
        if enable_audio_playback:
            self.audio_player = AudioPlayerAsync(
                sample_rate=self.config.audio.sample_rate,
                channels=self.config.audio.channels,
            )
            await self.audio_player.start()
        else:
            await self.audio_player.start()

        logger.info("Waiting for connection...")
        await self._connected.wait()
        logger.info("✓ Audio sending started")
        try:
            while self.is_running:
                data, overflowed = await self.audio_player.read(frames=self.read_size)
                if overflowed:
                    logger.warning("⚠️ Audio recorder overflowed")
                await self.connection.input_audio_buffer.append(
                    audio=encode_audio_to_base64(data)
                )
                if enable_audio_playback and self.audio_player:
                    await self.audio_player.play(data)
                await asyncio.sleep(0)  # 让出控制权
        except asyncio.CancelledError:
            logger.info("Audio sending task cancelled")
        finally:
            logger.info("✓ Audio sending stopped")
            if self.audio_player:
                await self.audio_player.stop()

    async def run(self, enable_audio_playback: bool = True) -> None:
        # 不要裸 gather；一个崩了要能停掉另一个
        self.is_running = True  # 在启动任务前设置运行状态
        t_conn = asyncio.create_task(self.connect(), name="connect")
        t_send = asyncio.create_task(
            self.send_audio_loop(enable_audio_playback), name="send_audio"
        )

        done, pending = await asyncio.wait(
            {t_conn, t_send}, return_when=asyncio.FIRST_EXCEPTION
        )
        for t in pending:
            logger.info(f"Cancelling task: {t.get_name()}")
            t.cancel()
        for t in done:
            try:
                t.result()
            except Exception as e:
                logger.error(f"Task {t.get_name()} failed: {e}", exc_info=True)

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
            close_result = self.audio_player.close()
            if asyncio.iscoroutine(close_result):
                await close_result
