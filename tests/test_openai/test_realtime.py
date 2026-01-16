import asyncio
import base64
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from openai import AsyncOpenAI
from openai.types.realtime.conversation_item import ConversationItem
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_session_create_request import (
    RealtimeSessionCreateRequest,
)


def load_24k_base64_audio(input_path: Path) -> str:
    audio, sr = librosa.load(input_path, sr=24000, mono=True)
    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_b64


async def main(input_path: Path):
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"Using OPENAI_BASE_URL={client.base_url}, OPENAI_API_KEY={client.api_key}")

    async with client.realtime.connect(model="gpt-4o") as connection:
        await connection.session.update(
            session=RealtimeSessionCreateRequest(
                type="realtime",
                output_modalities=["text"],
            )
        )
        audio_b64 = load_24k_base64_audio(input_path)
        await connection.conversation.item.create(
            item=RealtimeConversationItemUserMessage(
                role="user",
                type="message",
                content=[Content(type="input_audio", audio=audio_b64)],
            )
        )
        await connection.response.create()

        async for event in connection:
            print(event)
            if event.type == "response.output_text.delta":
                print(event.delta, flush=True, end="")

            elif event.type == "response.output_text.done":
                print()

            elif event.type == "response.done":
                break


asyncio.run(main(Path("data-bin/雷总.wav")))
