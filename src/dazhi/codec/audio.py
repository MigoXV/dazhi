"""音频编解码模块"""
import base64
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore

# 音频配置常量
CHANNELS = 1
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 48000


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


class AudioRecorder:
    """音频录制器"""

    def __init__(
        self,
        channels: int = CHANNELS,
        sample_rate: int = SAMPLE_RATE,
        dtype: str = "int16",
    ):
        """
        初始化音频录制器
        
        Args:
            channels: 音频通道数，默认为1（单声道）
            sample_rate: 采样率，默认为16000
            dtype: 数据类型，默认为"int16"
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.stream = None

    def start(self):
        """启动音频流"""
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
        )
        self.stream.start()

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        """
        读取音频数据
        
        Args:
            frames: 要读取的帧数
            
        Returns:
            (音频数据数组, 是否溢出)
        """
        if self.stream is None:
            raise RuntimeError("音频流未启动，请先调用 start()")
        return self.stream.read(frames)

    def read_available(self) -> int:
        """获取可读取的帧数"""
        if self.stream is None:
            return 0
        return self.stream.read_available

    def stop(self):
        """停止音频流"""
        if self.stream:
            self.stream.stop()

    def close(self):
        """关闭音频流"""
        if self.stream:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
        self.close()


def encode_audio_to_base64(audio_data: bytes | np.ndarray) -> str:
    """
    将音频数据编码为 base64 字符串
    
    Args:
        audio_data: 音频字节数据或 numpy 数组
        
    Returns:
        base64 编码的字符串
    """
    if isinstance(audio_data, np.ndarray):
        audio_data = audio_data.tobytes()
    return base64.b64encode(cast(Any, audio_data)).decode("utf-8")


def decode_audio_from_base64(base64_str: str) -> bytes:
    """
    从 base64 字符串解码音频数据
    
    Args:
        base64_str: base64 编码的字符串
        
    Returns:
        解码后的音频字节数据
    """
    return base64.b64decode(base64_str)


def query_audio_devices():
    """
    查询可用的音频设备
    
    Returns:
        音频设备信息
    """
    return sd.query_devices()
