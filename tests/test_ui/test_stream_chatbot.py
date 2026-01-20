import datetime
import queue
import random
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from gradio import ChatMessage
from typing import Dict
from dazhi.ui.chatbot import StreamChatbot


@dataclass
class StreamState:
    audio_stream: Optional[np.ndarray] = None
    transcript_text: str = ""
    last_audio_len: float = 0.0
    time_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))
    time_thread_started: bool = False
    pending_response: bool = False
    queue_timeout: float = 0.2
    time_interval_min: float = 1.0
    time_interval_max: float = 2.0


def _time_worker(
    time_queue: queue.Queue, min_interval: float, max_interval: float
) -> None:
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            time_queue.put_nowait(current_time)
        except queue.Full:
            try:
                time_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                time_queue.put_nowait(current_time)
            except queue.Full:
                pass
        time.sleep(random.uniform(min_interval, max_interval))


def _start_time_thread(state: StreamState) -> None:
    thread = threading.Thread(
        target=_time_worker,
        args=(state.time_queue, 1, 4),
        name="time-worker",
        daemon=True,
    )
    thread.start()
    state.time_thread_started = True


def _ensure_state(state: Optional[StreamState]) -> StreamState:
    if state is None or not isinstance(state, StreamState):
        state = StreamState()
    if not state.time_thread_started:
        _start_time_thread(state)
    return state


def transcribe_audio(
    state: Optional[StreamState],
    new_chunk: Optional[Tuple[int, np.ndarray]],
    history: List[Dict],
):
    """流式转写音频并更新用户消息"""
    state = _ensure_state(state)
    if new_chunk is not None:
        sr, audio_data = new_chunk
        state.audio_stream = (
            np.concatenate((state.audio_stream, audio_data), axis=-1)
            if state.audio_stream is not None
            else audio_data
        )
        state.last_audio_len = state.audio_stream.shape[-1] / sr

    try:
        current_time = state.time_queue.get(timeout=state.queue_timeout)
    except queue.Empty:
        return state, history

    new_line = f"{current_time} - Audio length: {state.last_audio_len:.2f}"
    if state.transcript_text:
        state.transcript_text = f"{state.transcript_text}\n{new_line}"
    else:
        state.transcript_text = new_line

    if history and history[-1]["role"] == "user":
        history[-1]["content"] = state.transcript_text
    else:
        history.append({"role": "user", "content": state.transcript_text})

    state.pending_response = True
    return state, history


def stream_text(
    text: str, min_delay: float = 0.01, max_delay: float = 0.04
) -> Iterable[str]:
    for character in text:
        time.sleep(random.uniform(min_delay, max_delay))
        yield character


def stub_reply(state: Optional[StreamState], history: List[Dict]):
    """生成 stub 回复，用于驱动 StreamChatbot.handle_bot 的流式输出"""
    state = _ensure_state(state)
    last_user_message = ""
    if history:
        last_user_message = history[-1]["content"]
        response = "今天的天气真好" * 10 + f"，你刚才说的是：{last_user_message}"
    else:
        response = "未收到用户消息。"
    return stream_text(response), state, history


if __name__ == "__main__":
    chatbot = StreamChatbot(
        audio_stream_handler=transcribe_audio,
        bot_stream_handler=stub_reply,
    )
    chatbot.launch()
