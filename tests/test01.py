from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import gradio as gr
from openai import AsyncOpenAI

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
    client = AsyncOpenAI()

    audio_b64 = gradio_audio_to_base64_pcm16(audio_in)

    out_audio = bytearray()
    user_tx_parts: List[str] = []
    asr_tx_parts: List[str] = []
    out_text_parts: List[str] = []

    async with client.realtime.connect(model=model) as connection:  # :contentReference[oaicite:3]{index=3}
        # 会话配置：同时要 audio + text；输入转写打开；音频格式用 24k PCM
        await connection.session.update(
            session={
                "type": "realtime",
                # 同时输出音频和文本（如果你只想要音频，就只留 ["audio"]）
                "output_modalities": ["audio", "text"],
                "instructions": system_prompt,
                # 关闭 VAD（可选；这里我们是整段音频一次性发送，其实不强依赖）
                "turn_detection": None,  # 文档里讲“turn_detection: null”用于手动控制 :contentReference[oaicite:4]{index=4}
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": TARGET_SR},
                        "transcription": {"model": input_transcribe_model},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": TARGET_SR},
                        "voice": voice,
                    },
                },
            }
        )

        # 把“文本 + 音频”作为一次 user message 发进去（最省事的 dummy 做法）
        content = []
        if user_text.strip():
            content.append({"type": "input_text", "text": user_text.strip()})
        content.append({"type": "input_audio", "audio": audio_b64})  # 文档支持整段 input_audio :contentReference[oaicite:5]{index=5}

        await connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": content}
        )

        # 触发模型生成
        await connection.response.create()

        async for event in connection:
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

            # 错误事件（SDK 不会直接 raise，需要你自己处理）:contentReference[oaicite:6]{index=6}
            elif et == "error":
                err = getattr(event, "error", None)
                msg = getattr(err, "message", "unknown error") if err else "unknown error"
                raise RuntimeError(f"Realtime error: {msg}")

            elif et == "response.done":
                break

    return RealtimeResult(
        audio_pcm16=bytes(out_audio),
        user_transcript="".join(user_tx_parts).strip(),
        assistant_audio_transcript="".join(asr_tx_parts).strip(),
        assistant_text="".join(out_text_parts).strip(),
    )


def pcm16_bytes_to_gradio_audio(pcm16: bytes, sr: int = TARGET_SR) -> Optional[Tuple[int, np.ndarray]]:
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
    system_prompt = gr.Textbox(label="System Prompt", lines=2, value="你是一个简洁、口语化的中文语音助手。")

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
