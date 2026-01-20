"""音频编解码工具函数"""
import base64
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore


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
