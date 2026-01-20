import datetime
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import gradio as gr
import numpy as np

@dataclass
class StreamState:
    audio_stream: Optional[np.ndarray] = None
    transcript_text: str = ""
    last_audio_len: float = 0.0
    time_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))
    time_thread_started: bool = False


def _time_worker(time_queue: queue.Queue, interval_seconds: float) -> None:
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
        time.sleep(random.uniform(1, 2))


def _start_time_thread(state: StreamState) -> None:
    thread = threading.Thread(
        target=_time_worker,
        args=(state.time_queue, 3),
        name="time-worker",
        daemon=True,
    )
    thread.start()
    state.time_thread_started = True


def transcribe(state: Optional[StreamState], new_chunk: Optional[Tuple[int, np.ndarray]]):
    """打印音频流的形状和数据类型，并返回当前时间和日期"""
    if state is None:
        state = StreamState()
    if not state.time_thread_started:
        _start_time_thread(state)

    if new_chunk is not None:
        sr, audio_data = new_chunk
        state.audio_stream = (
            np.concatenate((state.audio_stream, audio_data), axis=-1)
            if state.audio_stream is not None
            else audio_data
        )
        state.last_audio_len = state.audio_stream.shape[-1] / sr

    try:
        current_time = state.time_queue.get(timeout=0.05)
    except queue.Empty:
        return state, state.transcript_text

    new_line = f"{current_time} - Audio length: {state.last_audio_len:.2f}"
    if state.transcript_text:
        state.transcript_text = f"{state.transcript_text}\n{new_line}"
    else:
        state.transcript_text = new_line
    return state, state.transcript_text


audio_component = gr.Audio(
    sources=["microphone"],
    streaming=True,
    type="numpy",
    label="Stream Audio Input",
    every=0.2,
    show_label=True,
    show_download_button=True,
)
transcript_output = gr.Textbox(
    label="Transcription",
    lines=12,
    max_lines=20,
    show_copy_button=True,
)
demo = gr.Interface(
    transcribe,
    ["state", audio_component],
    ["state", transcript_output],
    live=True,
    api_name="predict",
)
demo.launch()
