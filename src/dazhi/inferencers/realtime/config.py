"""实时推理配置模块"""

import os
from dataclasses import dataclass, field
from typing import Optional

from openai.types.realtime import RealtimeAudioConfig

from dazhi.codec import CHANNELS, SAMPLE_RATE


@dataclass
class OpenAIConfig:
    """OpenAI客户端配置"""

    base_url: str | None = None
    api_key: str | None = None

    def __post_init__(self):
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class RealtimeConnectionConfig:
    """实时连接配置"""

    model: str = "transcribe"
    ssl_verify: bool = False


@dataclass
class RealtimeSessionConfig:
    """实时会话配置"""

    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    audio: Optional[RealtimeAudioConfig] = None


@dataclass
class AudioConfig:
    """音频配置"""

    channels: int = CHANNELS
    sample_rate: int = SAMPLE_RATE
    read_interval: float = 0.02  # 20ms


@dataclass
class RealtimeConfig:
    """实时推理配置 - 组合所有配置"""

    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    connection: RealtimeConnectionConfig = field(
        default_factory=RealtimeConnectionConfig
    )
    session: RealtimeSessionConfig = field(default_factory=RealtimeSessionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
