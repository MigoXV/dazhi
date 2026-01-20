"""音频播放器实现"""

import numpy as np
import sounddevice as sd  # type: ignore

from .config import CHANNELS, OUTPUT_SAMPLE_RATE


class AudioPlayerAsync:
    """异步音频播放器"""

    def __init__(
        self,
        channels: int = CHANNELS,
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        dtype: str = "int16",
    ):
        """
        初始化音频播放器
        
        Args:
            channels: 音频通道数，默认为1（单声道）
            sample_rate: 采样率，默认为48000
            dtype: 数据类型，默认为"int16"
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.frame_count = 0

        # 创建音频输出流
        self.output_stream = sd.OutputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
        )
        self.output_stream.start()

    def reset_frame_count(self):
        """重置帧计数"""
        self.frame_count = 0

    def add_data(self, data: bytes):
        """
        添加音频数据并播放
        
        Args:
            data: 音频字节数据
        """
        # 将字节数据转换为 numpy 数组并播放
        audio_array = np.frombuffer(data, dtype=np.int16)
        # 重塑为二维数组 (frames, channels)
        audio_array = audio_array.reshape(-1, self.channels)
        self.output_stream.write(audio_array)
        self.frame_count += len(audio_array)

    def close(self):
        """关闭音频流"""
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
