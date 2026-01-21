from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from dazhi.inferencers.realtime.inferencer import RealtimeInferencer

TARGET_SR = 24000  # Realtime 文档示例常用 24k PCM :contentReference[oaicite:2]{index=2}


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    # (n, ch) -> mono
    y_f = y.astype(np.float32).mean(axis=1)
    return y_f.astype(y.dtype)


def _ensure_int16(y: np.ndarray) -> np.ndarray:
    if y.dtype == np.int16:
        return y
    # 常见情况：float32 in [-1, 1]
    if np.issubdtype(y.dtype, np.floating):
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).round().astype(np.int16)
    # 兜底：按比例压到 int16
    y_f = y.astype(np.float32)
    mx = float(np.max(np.abs(y_f))) if y_f.size else 0.0
    if mx > 0:
        y_f = y_f / mx
    return (np.clip(y_f, -1.0, 1.0) * 32767.0).round().astype(np.int16)


def _resample_linear_int16(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or y.size == 0:
        return y
    n_in = y.shape[0]
    dur = n_in / float(sr_in)
    n_out = max(1, int(round(dur * sr_out)))

    x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)

    y_f = np.interp(x_new, x_old, y.astype(np.float32))
    return np.clip(np.round(y_f), -32768, 32767).astype(np.int16)


def gradio_audio_to_base64_pcm16(audio: Tuple[int, np.ndarray]) -> str:
    sr, y = audio
    y = _to_mono(y)
    y = _ensure_int16(y)
    y = _resample_linear_int16(y, sr, TARGET_SR)
    pcm_bytes = y.tobytes()  # little-endian int16
    return base64.b64encode(pcm_bytes).decode("ascii")


@dataclass
class RealtimeResult:
    audio_pcm16: bytes
    user_transcript: str
    assistant_audio_transcript: str
    assistant_text: str


async def realtime_s2s_once(
    audio_in: Tuple[int, np.ndarray],
    user_text: str,
    system_prompt: str,
    model: str = "gpt-realtime",
    voice: str = "marin",
    input_transcribe_model: str = "whisper-1",
) -> RealtimeResult:
    audio_b64 = gradio_audio_to_base64_pcm16(audio_in)

    out_audio = bytearray()
    user_tx_parts: List[str] = []
    asr_tx_parts: List[str] = []
    out_text_parts: List[str] = []

    base_url=os.getenv("OPENAI_BASE_URL")
    api_key=os.getenv("OPENAI_API_KEY")
    print(f"Using OPENAI_BASE_URL={base_url}, OPENAI_API_KEY={api_key}")

    inferencer = RealtimeInferencer(
        base_url=base_url,
        api_key=api_key,
        model=model,
        voice=voice,
        input_transcribe_model=input_transcribe_model,
        sample_rate=TARGET_SR,
    )

    # 把“文本 + 音频”作为一次 user message 发进去（最省事的 dummy 做法）
    content = []
    if user_text.strip():
        content.append({"type": "input_text", "text": user_text.strip()})
    content.append(
        {"type": "input_audio", "audio": audio_b64}
    )  # 文档支持整段 input_audio :contentReference[oaicite:5]{index=5}

    async for event in inferencer.infer(
        items=[{"type": "message", "role": "user", "content": content}],
        system_prompt=system_prompt,
    ):
        et = event.type

        # 输入音频转写（用户说了什么）
        if et == "conversation.item.input_audio_transcription.completed":
            # 不同 SDK 版本字段可能有差异，这里做个 getattr 兜底
            tx = getattr(event, "transcript", "") or ""
            if tx:
                user_tx_parts.append(tx)

        # 助手输出文本（如果 output_modalities 里带了 text）
        elif et == "response.output_text.delta":
            out_text_parts.append(getattr(event, "delta", "") or "")

        # 助手输出音频的“转写文本”（边说边给字幕）
        elif et == "response.output_audio_transcript.delta":
            asr_tx_parts.append(getattr(event, "delta", "") or "")

        # 助手输出音频数据（base64 PCM16 chunk）
        elif et == "response.output_audio.delta":
            b64 = getattr(event, "delta", "") or ""
            if b64:
                out_audio.extend(base64.b64decode(b64))

        elif et == "response.done":
            break

    return RealtimeResult(
        audio_pcm16=bytes(out_audio),
        user_transcript="".join(user_tx_parts).strip(),
        assistant_audio_transcript="".join(asr_tx_parts).strip(),
        assistant_text="".join(out_text_parts).strip(),
    )


def pcm16_bytes_to_gradio_audio(
    pcm16: bytes, sr: int = TARGET_SR
) -> Optional[Tuple[int, np.ndarray]]:
    if not pcm16:
        return None
    y = np.frombuffer(pcm16, dtype=np.int16)
    return (sr, y)


async def gradio_handler(audio, user_text, system_prompt):
    if audio is None:
        return None, "", "", ""

    r = await realtime_s2s_once(
        audio_in=audio,
        user_text=user_text or "",
        system_prompt=system_prompt or "你是一个简洁、口语化的中文语音助手。",
    )
    audio_out = pcm16_bytes_to_gradio_audio(r.audio_pcm16, TARGET_SR)

    return audio_out, r.user_transcript, r.assistant_audio_transcript, r.assistant_text


with gr.Blocks(title="Realtime S2S Dummy Demo (OpenAI SDK + Gradio)") as demo:
    gr.Markdown("录一段音频 → Realtime → 返回助手语音 + 转写/文本")

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone"],
            type="numpy",  # gradio 会给 (sr, np.ndarray)，常见是 int16 :contentReference[oaicite:7]{index=7}
            label="你的语音输入",
        )
        audio_out = gr.Audio(type="numpy", label="助手语音输出")

    user_text = gr.Textbox(label="可选：额外文本（会和语音一起发给模型）", lines=2)
    system_prompt = gr.Textbox(
        label="System Prompt", lines=2, value="你是一个简洁、口语化的中文语音助手。"
    )

    btn = gr.Button("发送")
    user_tx = gr.Textbox(label="用户转写（input_audio transcription）", lines=2)
    asst_tx = gr.Textbox(label="助手音频转写（output_audio_transcript）", lines=2)
    asst_text = gr.Textbox(label="助手文本输出（output_text）", lines=3)

    btn.click(
        gradio_handler,
        inputs=[audio_in, user_text, system_prompt],
        outputs=[audio_out, user_tx, asst_tx, asst_text],
    )

demo.queue().launch()
