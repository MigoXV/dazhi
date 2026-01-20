import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from dazhi.codec.recorders import QueueAudioRecorder
from dazhi.handlers import GradioEventHandler
from dazhi.inferencers.realtime.inferencer import RealtimeConfig, RealtimeInferencer

logger = logging.getLogger(__name__)


@dataclass
class ChatbotState:
    """聊天机器人状态类"""

    realtime_config: RealtimeConfig = field(default_factory=RealtimeConfig)
    recorder: QueueAudioRecorder = field(default_factory=QueueAudioRecorder)
    history: List[Dict[str, str]] = field(default_factory=list)
    _inferencer_task: Optional[asyncio.Task] = None

    def __post_init__(self):
        event_handler = GradioEventHandler(chatbot_history=self.history)
        self.inferencer = RealtimeInferencer(
            self.realtime_config,
            event_handler=event_handler,
            audio_recorder=self.recorder,
        )

    async def start_inferencer(self):
        """启动 inferencer 作为后台任务"""
        if self._inferencer_task is None or self._inferencer_task.done():
            self._inferencer_task = asyncio.create_task(
                self.inferencer.run(enable_audio_playback=False)
            )
            # 等待连接建立
            await self.inferencer._connected.wait()
            logger.info("Inferencer background task started")

    async def stop_inferencer(self):
        """停止 inferencer 后台任务"""
        if self.inferencer:
            await self.inferencer.stop()
        if self._inferencer_task:
            try:
                await asyncio.wait_for(self._inferencer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Inferencer task did not stop in time, cancelling...")
                self._inferencer_task.cancel()
            except Exception as e:
                logger.error(f"Error stopping inferencer task: {e}")


class StreamChatbot:
    """流式聊天机器人界面类"""

    def __init__(self, realtime_config: RealtimeConfig):
        self.demo = None
        self._build_interface()

        self.realtime_config = realtime_config

    def _build_interface(self):
        """构建 Gradio 界面"""
        # 构建ui
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        type="messages",
                        allow_tags=True,
                        height=600,
                        group_consecutive_messages=False,
                    )
                with gr.Column(scale=1):
                    audio = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Stream Audio Input",
                        every=0.5,
                        show_label=True,
                        show_download_button=True,
                    )
            # 绑定事件
            chat_states = gr.State(value=None)
            audio_stream_event = audio.stream(
                self.handle_audio_stream,
                inputs=[audio, chat_states],
                outputs=[chatbot, chat_states],
            )
            # 添加清理事件处理
            demo.unload(lambda: self.handle_cleanup(chat_states.value))
        self.demo = demo

    def launch(self, **kwargs):
        self.demo.queue()
        return self.demo.launch(**kwargs)

    async def handle_audio_stream(
        self, audio_data: Tuple[int, np.ndarray], state: Optional[ChatbotState] = None
    ):
        state = await self.ensure_state_initialized(state)
        # 将音频数据送入处理器
        try:
            sr, audio = audio_data
            audio = audio.astype(np.float32) / 32768.0
            await state.recorder.put_audio(audio, sr)
        except Exception as e:
            logger.error(f"Failed to put audio data into recorder: {e}")

        return state.history, state

    async def ensure_state_initialized(
        self, state: Optional[ChatbotState]
    ) -> ChatbotState:
        """确保聊天机器人状态已初始化"""
        if state is None:
            state = ChatbotState(realtime_config=self.realtime_config)
            await state.start_inferencer()
            logger.info("Initialized new ChatbotState")
        return state

    async def handle_cleanup(self, state: Optional[ChatbotState]):
        """处理资源清理"""
        if state is not None:
            logger.info("Cleaning up ChatbotState resources...")
            await state.stop_inferencer()
            logger.info("ChatbotState resources cleaned up")
