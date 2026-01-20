#!/usr/bin/env python3
"""
Simplified Realtime API app - immediately starts audio streaming and receiving results.
使用 RealtimeInferencer 推理器的简化版本。
"""
import asyncio

from dazhi.codec.audio import AudioRecorder
from dazhi.inferencers.realtime.inferencer import (
    RealtimeConfig,
    RealtimeInferencer,
)


async def main():
    config = RealtimeConfig(
        model="transcrib",
        output_modalities=["text"],
    )
    inferencer = RealtimeInferencer(config)
    recorder = AudioRecorder(
        channels=config.channels,
        sample_rate=config.sample_rate,
    )
    await inferencer.run(audio_recorder=recorder)


if __name__ == "__main__":
    asyncio.run(main())
