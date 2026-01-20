import logging
import queue
from abc import ABC, abstractmethod

import asyncio
import librosa
import numpy as np
import sounddevice as sd  # type: ignore

from .config import CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioRecorder(ABC):
    """音频录制器抽象基类"""

    @abstractmethod
    async def start(self) -> None:
        """启动音频流"""
        pass

    @abstractmethod
    async def read(self, frames: int) -> tuple[np.ndarray, bool]:
        """
        读取音频数据

        Args:
            frames: 要读取的帧数

        Returns:
            (音频数据数组, 是否溢出)
        """
        pass

    @abstractmethod
    async def read_available(self) -> int:
        """获取可读取的帧数"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止音频流"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭音频流"""
        pass

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.stop()
        await self.close()


class SDAudioRecorder(AudioRecorder):
    """基于 sounddevice 的音频录制器实现"""

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

    async def start(self) -> None:
        """启动音频流"""
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
        )
        self.stream.start()

    async def read(self, frames: int) -> tuple[np.ndarray, bool]:
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

    async def read_available(self) -> int:
        """获取可读取的帧数"""
        if self.stream is None:
            return 0
        return self.stream.read_available

    async def stop(self) -> None:
        """停止音频流"""
        if self.stream:
            self.stream.stop()

    async def close(self) -> None:
        """关闭音频流"""
        if self.stream:
            self.stream.close()
            self.stream = None


class QueueAudioRecorder(AudioRecorder):
    """基于队列的音频录制器实现，支持注入音频数据"""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        dtype: str = "float32",
    ):
        """
        初始化基于队列的音频录制器

        Args:
            channels: 音频通道数，默认为1（单声道）
            sample_rate: 采样率，默认为16000
            dtype: 数据类型，默认为"float32"
        """
        self.channels = CHANNELS
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.queue = asyncio.Queue()
        self.is_running = False
        self._buffer = np.array([], dtype=self.dtype)

    async def start(self) -> None:
        """启动队列录制器"""
        self.is_running = True

    async def put_audio(self, audio: np.ndarray, sr: int) -> None:
        """
        注入音频数据到队列

        Args:
            audio: 音频数据数组 (float32)
        """
        # 确保数据类型正确
        if audio.dtype != self.dtype:
            raise ValueError(f"音频数据类型应为 {self.dtype}，但收到 {audio.dtype}")
        # 确保通道数正确
        if audio.ndim != 1:
            raise ValueError("音频数据应为单声道一维数组")
        # 重采样到目标采样率
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        await self.queue.put(audio)

    async def read(self, frames: int) -> tuple[np.ndarray, bool]:
        """
        从队列中读取音频数据

        Args:
            frames: 要读取的帧数

        Returns:
            (音频数据数组, 是否溢出)
        """
        if not self.is_running:
            raise RuntimeError("录制器未启动，请先调用 start()")

        # 从缓冲区和队列中收集数据直到达到所需帧数
        while len(self._buffer) < frames:
            try:
                audio_data = await self.queue.get()
                self._buffer = np.concatenate([self._buffer, audio_data])
            except Exception as e:
                logger.error(f"Error while getting audio data from queue: {e}")
                break
        # 如果缓冲区有足够数据
        if len(self._buffer) >= frames:
            result = self._buffer[:frames]
            self._buffer = self._buffer[frames:]
            if result.dtype != np.int16:
                result = (result * 32767).astype(np.int16)
            return result, False
        # 缓冲区数据不足
        elif len(self._buffer) > 0:
            logger.warning("音频数据不足，返回部分数据并用零填充")
            result = self._buffer
            self._buffer = np.array([], dtype=self.dtype).reshape(0, self.channels)
            # 用零填充不足的部分
            padding = np.zeros((frames - len(result), self.channels), dtype=self.dtype)
            result = np.concatenate([result, padding])
            return result, False
        else:
            # 没有数据，返回空数据
            return np.zeros((frames, self.channels), dtype=self.dtype), False

    async def read_available(self) -> int:
        """
        获取队列中所有音频块并拼接为一整块音频

        Returns:
            拼接后的完整音频数据数组 (float32)
        """
        audio_chunks = []

        # 如果有缓冲区数据，先加入
        if len(self._buffer) > 0:
            audio_chunks.append(self._buffer)
            self._buffer = np.array([], dtype=self.dtype).reshape(0, self.channels)

        # 从队列中取出所有数据
        while not self.queue.empty():
            try:
                audio_data = await self.queue.get()
                audio_chunks.append(audio_data)
            except Exception as e:
                logger.error(f"Error while getting audio data from queue: {e}")
                break

        # 拼接所有音频块
        if audio_chunks:
            return np.vstack(audio_chunks)
        else:
            return np.array([], dtype=self.dtype).reshape(0, self.channels)

    async def stop(self) -> None:
        """停止队列录制器"""
        self.is_running = False

    async def close(self) -> None:
        """关闭队列录制器并清空队列"""
        self.is_running = False
        self._buffer = np.array([], dtype=self.dtype).reshape(0, self.channels)
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception as e:
                logger.error(f"Error while clearing audio data from queue: {e}")
                break
