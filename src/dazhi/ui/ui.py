"""Gradio UI for Voice + MCP Inferencer
实现语音输入 + MCP 工具调用的 Chatbot 界面
边录边转模式：使用后台线程 + 队列实现实时转写
"""

import asyncio
import json
import os
import queue
import ssl
import threading
from dataclasses import dataclass, field
from typing import Generator, Optional

import gradio as gr
import httpx
import numpy as np
from openai import AsyncOpenAI

from dazhi.codec.audio import SAMPLE_RATE, encode_audio_to_base64
from dazhi.inferencers.mcp.inferencer import (
    LLMConfig,
    MCPClient,
    MCPConfig,
)


@dataclass
class SessionState:
    """每个用户会话的状态 - 使用后台线程 + 队列模式"""
    is_active: bool = False
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    results_queue: queue.Queue = field(default_factory=queue.Queue)
    llm_request_queue: queue.Queue = field(default_factory=queue.Queue)
    llm_response_queue: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: Optional[threading.Thread] = None
    llm_worker_thread: Optional[threading.Thread] = None
    current_transcript: str = ""
    audio_buffer: list = field(default_factory=list)
    error_message: str = ""


class VoiceMCPGradioApp:
    """语音 + MCP Gradio 应用"""

    def __init__(
        self,
        mcp_config: MCPConfig,
        llm_config: LLMConfig | None = None,
    ):
        self.mcp_config = mcp_config
        self.llm_config = llm_config or LLMConfig()
        
        # 共享的客户端（线程安全）
        self.mcp_client: MCPClient | None = None
        self.llm_client: AsyncOpenAI | None = None
        self.messages: list[dict] = []
        self.is_connected: bool = False

        # Realtime API 配置
        self.realtime_base_url = os.getenv("OPENAI_BASE_URL")
        self.realtime_api_key = os.getenv("OPENAI_API_KEY")
        self.realtime_model = "transcribe"

    def _run_async(self, coro):
        """运行异步协程"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def connect(self) -> str:
        """连接 MCP 和 LLM 服务"""
        async def _connect():
            try:
                # 初始化 MCP 客户端
                self.mcp_client = MCPClient(self.mcp_config)
                await self.mcp_client.connect()

                # 初始化 LLM 客户端
                self.llm_client = AsyncOpenAI(
                    base_url=self.llm_config.base_url,
                    api_key=self.llm_config.api_key,
                    http_client=httpx.AsyncClient(verify=False),
                )

                # 初始化消息历史
                self.messages = [
                    {"role": "system", "content": self.llm_config.system_prompt}
                ]

                self.is_connected = True
                tools = [t.name for t in self.mcp_client.tools]
                return f"✅ 连接成功！\n可用工具: {', '.join(tools)}"

            except Exception as e:
                self.is_connected = False
                return f"❌ 连接失败: {e}"
        
        return self._run_async(_connect())

    def disconnect(self) -> str:
        """断开连接"""
        async def _disconnect():
            try:
                if self.mcp_client:
                    await self.mcp_client.disconnect()
                self.is_connected = False
                self.mcp_client = None
                self.llm_client = None
                return "已断开连接"
            except Exception as e:
                return f"断开连接时出错: {e}"
        
        return self._run_async(_disconnect())

    def process_text(
        self,
        message: str,
        history: list,
    ) -> Generator:
        """处理文本输入（流式输出）"""
        if not message.strip():
            yield history
            return

        if not self.is_connected:
            history = history + [{"role": "assistant", "content": "❌ 请先点击「连接服务」"}]
            yield history
            return

        # 添加用户消息到历史
        history = history + [{"role": "user", "content": message}]
        yield history

        # 添加到内部消息列表（加 /no_think 关闭思考）
        self.messages.append({"role": "user", "content": message + " /no_think"})

        # 获取 MCP 工具
        tools = self.mcp_client.get_tools_for_openai() if self.mcp_client else []

        async def _process():
            try:
                # 调用 LLM（流式）
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=self.messages,
                    tools=tools if tools else None,
                    stream=True,
                )

                content_chunks = []
                tool_calls = []

                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    if delta.content:
                        content_chunks.append(delta.content)

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            while len(tool_calls) <= tc.index:
                                tool_calls.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""},
                                })
                            target = tool_calls[tc.index]
                            if tc.id:
                                target["id"] = tc.id
                            if tc.function and tc.function.name:
                                target["function"]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                target["function"]["arguments"] += tc.function.arguments

                return "".join(content_chunks), tool_calls

            except Exception as e:
                return f"❌ 出错: {e}", []

        content, tool_calls = self._run_async(_process())
        
        # 保存助手消息
        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self.messages.append(assistant_msg)

        # 显示助手回复
        history = history + [{"role": "assistant", "content": content}]
        yield history

        # 处理工具调用
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args_str = tc["function"]["arguments"]
                tool_args = json.loads(tool_args_str) if tool_args_str else {}

                # 显示工具调用
                history[-1]["content"] += f"\n\n🔧 **调用工具**: \`{tool_name}\`"
                yield history

                # 执行工具
                async def _call_tool():
                    try:
                        result = await self.mcp_client.call_tool(tool_name, tool_args)
                        return result
                    except Exception as e:
                        return f"工具调用失败: {e}"

                result = self._run_async(_call_tool())
                result_preview = result[:300] + "..." if len(result) > 300 else result
                history[-1]["content"] += f"\n📋 **结果**: {result_preview}"
                yield history

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # 再次调用 LLM
            async def _final_response():
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=self.messages,
                    stream=True,
                )
                chunks = []
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                return "".join(chunks)

            final_content = self._run_async(_final_response())
            self.messages.append({"role": "assistant", "content": final_content})
            history = history + [{"role": "assistant", "content": final_content}]
            yield history

    def _websocket_worker(self, state: SessionState):
        """后台线程：管理 WebSocket 连接，发送音频并接收转写结果"""
        
        async def _run():
            try:
                # 创建 Realtime 客户端
                client = AsyncOpenAI(
                    base_url=self.realtime_base_url,
                    api_key=self.realtime_api_key,
                )
                
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                async with client.realtime.connect(
                    model=self.realtime_model,
                    websocket_connection_options={"ssl": ssl_context},
                ) as conn:
                    # 配置会话
                    await conn.session.update(
                        session={
                            "output_modalities": ["text"],
                        }
                    )
                    print("[DEBUG] WebSocket 连接已建立，Session 已配置")
                    state.results_queue.put(("connected", None))
                    
                    async def send_audio():
                        """从队列读取音频并发送 - 服务端自动 VAD"""
                        while not state.stop_event.is_set():
                            try:
                                audio_base64 = state.audio_queue.get(timeout=0.1)
                                if audio_base64 == "STOP_SESSION":
                                    print("[DEBUG] 收到停止信号")
                                    break
                                await conn.input_audio_buffer.append(audio=audio_base64)
                            except queue.Empty:
                                await asyncio.sleep(0.01)
                            except Exception as e:
                                print(f"[DEBUG] 发送音频错误: {e}")
                                break
                    
                    async def receive_events():
                        """接收转写事件 - 持续模式"""
                        try:
                            async for event in conn:
                                if state.stop_event.is_set():
                                    break
                                    
                                event_type = event.type
                                
                                if event_type == "response.output_audio_transcript.delta":
                                    # 每个 delta 都是完整可用的结果（服务端 VAD 切分）
                                    transcript = event.delta
                                    print(f"[DEBUG] 收到转写结果: {transcript}")
                                    state.results_queue.put(("transcript", transcript))
                                    
                                elif event_type == "response.done":
                                    print("[DEBUG] 响应完成，等待下一个音频块...")
                                    state.results_queue.put(("done", None))
                                    # 不 break，继续等待下一个音频块的转写
                                    
                                elif event_type == "error":
                                    print(f"[DEBUG] 错误事件: {event}")
                                    state.results_queue.put(("error", str(event)))
                                    # 错误时也不退出，让上层处理
                        except Exception as e:
                            print(f"[DEBUG] 接收事件错误: {e}")
                            state.results_queue.put(("error", str(e)))
                    
                    # 并行运行发送和接收
                    await asyncio.gather(
                        send_audio(),
                        receive_events(),
                        return_exceptions=True,
                    )
                    
            except Exception as e:
                print(f"[DEBUG] WebSocket worker 错误: {e}")
                state.results_queue.put(("error", str(e)))
            finally:
                state.results_queue.put(("closed", None))
                state.is_active = False
                print("[DEBUG] WebSocket worker 结束")
        
        # 在新的事件循环中运行
        asyncio.run(_run())

    def _llm_worker(self, state: SessionState):
        """后台线程：处理 LLM 请求"""
        while not state.stop_event.is_set():
            try:
                # 获取转写结果
                transcript = state.llm_request_queue.get(timeout=0.5)
                if transcript == "STOP":
                    break
                    
                print(f"[DEBUG] LLM worker 收到请求: '{transcript}'")
                
                # 添加用户消息到内部消息列表
                self.messages.append({"role": "user", "content": transcript + " /no_think"})
                
                # 获取 MCP 工具
                tools = self.mcp_client.get_tools_for_openai() if self.mcp_client else []
                
                # 调用 LLM
                async def _call_llm():
                    try:
                        response = await self.llm_client.chat.completions.create(
                            model=self.llm_config.model,
                            messages=self.messages,
                            tools=tools if tools else None,
                            stream=True,
                        )
                        
                        content_chunks = []
                        tool_calls = []
                        
                        async for chunk in response:
                            if not chunk.choices:
                                continue
                            delta = chunk.choices[0].delta
                            if delta is None:
                                continue
                            if delta.content:
                                content_chunks.append(delta.content)
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    while len(tool_calls) <= tc.index:
                                        tool_calls.append({
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": None, "arguments": ""},
                                        })
                                    target = tool_calls[tc.index]
                                    if tc.id:
                                        target["id"] = tc.id
                                    if tc.function and tc.function.name:
                                        target["function"]["name"] = tc.function.name
                                    if tc.function and tc.function.arguments:
                                        target["function"]["arguments"] += tc.function.arguments
                        
                        return "".join(content_chunks), tool_calls
                    except Exception as e:
                        return f"❌ LLM 出错: {e}", []
                
                content, tool_calls = self._run_async(_call_llm())
                
                # 保存助手消息
                assistant_msg = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                self.messages.append(assistant_msg)
                
                # 处理工具调用
                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc["function"]["name"]
                        tool_args_str = tc["function"]["arguments"]
                        import json
                        tool_args = json.loads(tool_args_str) if tool_args_str else {}
                        
                        async def _call_tool():
                            try:
                                result = await self.mcp_client.call_tool(tool_name, tool_args)
                                return result
                            except Exception as e:
                                return f"工具调用失败: {e}"
                        
                        result = self._run_async(_call_tool())
                        content += f"\n\n🔧 **调用工具**: `{tool_name}`\n📋 **结果**: {result[:300]}..."
                        
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })
                    
                    # 再次调用 LLM 获取最终回复
                    async def _final_response():
                        response = await self.llm_client.chat.completions.create(
                            model=self.llm_config.model,
                            messages=self.messages,
                            stream=True,
                        )
                        chunks = []
                        async for chunk in response:
                            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                chunks.append(chunk.choices[0].delta.content)
                        return "".join(chunks)
                    
                    final_content = self._run_async(_final_response())
                    self.messages.append({"role": "assistant", "content": final_content})
                    content = final_content
                
                # 将结果放入响应队列
                state.llm_response_queue.put(("response", transcript, content))
                print(f"[DEBUG] LLM worker 完成处理")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DEBUG] LLM worker 错误: {e}")
                import traceback
                traceback.print_exc()

    def _start_session(self, state: SessionState | None) -> tuple[SessionState, Optional[str]]:
        """启动新的转写会话"""
        # 停止旧会话
        if state is not None and state.is_active:
            state.stop_event.set()
            state.audio_queue.put("STOP_SESSION")
            state.llm_request_queue.put("STOP")
        
        # 创建新会话
        state = SessionState(is_active=True)
        
        # 启动 WebSocket worker 线程
        state.worker_thread = threading.Thread(
            target=self._websocket_worker,
            args=(state,),
            daemon=True,
        )
        state.worker_thread.start()
        
        # 启动 LLM worker 线程
        state.llm_worker_thread = threading.Thread(
            target=self._llm_worker,
            args=(state,),
            daemon=True,
        )
        state.llm_worker_thread.start()
        
        # 等待连接建立
        import time
        time.sleep(0.3)
        
        try:
            event_type, payload = state.results_queue.get_nowait()
            if event_type == "error":
                state.is_active = False
                return state, f"连接失败: {payload}"
            if event_type == "connected":
                print("[DEBUG] 会话已启动")
                return state, None
            # 放回队列
            state.results_queue.put((event_type, payload))
        except queue.Empty:
            pass
        
        return state, None

    def process_audio_stream(
        self,
        audio_chunk: tuple[int, np.ndarray] | tuple | None,
        history: list,
        state: SessionState | None,
    ) -> tuple[str, list, SessionState | None]:
        """处理流式音频 - 边录边转"""
        
        # 处理空音频数据
        if audio_chunk is None or len(audio_chunk) == 0:
            if state is not None and state.current_transcript:
                return f"🎤 {state.current_transcript}", gr.update(), state
            return "等待语音输入...", gr.update(), state

        if not self.is_connected:
            return "❌ 请先连接服务", gr.update(), state

        # 解包音频数据
        if len(audio_chunk) != 2:
            return "音频格式错误", gr.update(), state
            
        sample_rate, audio_array = audio_chunk

        try:
            # 转换音频格式
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            if audio_array.dtype != np.int16:
                if np.issubdtype(audio_array.dtype, np.floating):
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array = audio_array.astype(np.int16)

            # 重采样到 16000 Hz
            if sample_rate != SAMPLE_RATE:
                ratio = SAMPLE_RATE / sample_rate
                new_length = int(len(audio_array) * ratio)
                indices = np.linspace(0, len(audio_array) - 1, new_length).astype(int)
                audio_array = audio_array[indices].astype(np.int16)

            # 首次调用时启动会话
            if state is None or not state.is_active:
                state, error = self._start_session(state)
                if error:
                    return f"❌ {error}", history, state

            # 发送音频到队列（服务端自动 VAD）
            audio_base64 = encode_audio_to_base64(audio_array.tobytes())
            state.audio_queue.put(audio_base64)
            state.audio_buffer.append(audio_array)

            # 检查转写结果 - 每个 transcript delta 都是完整块（服务端 VAD）
            history_updated = False
            while True:
                try:
                    event_type, payload = state.results_queue.get_nowait()
                    if event_type == "transcript":
                        # 服务端 VAD 切分，每个 delta 都是完整句子
                        transcript = payload.strip()
                        if transcript:
                            state.current_transcript = transcript
                            print(f"[DEBUG] 收到完整块，发送给 LLM: '{transcript}'")
                            
                            # 添加用户消息到 history
                            history = history + [{"role": "user", "content": transcript}]
                            history_updated = True
                            
                            # 发送给 LLM worker 处理
                            state.llm_request_queue.put(transcript)
                        
                    elif event_type == "error":
                        state.error_message = payload
                    elif event_type == "done":
                        # 响应完成，不做任何事，等待新音频
                        print("[DEBUG] 响应完成")
                    elif event_type == "closed":
                        state.is_active = False
                except queue.Empty:
                    break
            
            # 检查 LLM 响应
            while True:
                try:
                    event_type, user_msg, assistant_msg = state.llm_response_queue.get_nowait()
                    if event_type == "response":
                        # 添加助手回复到 history
                        history = history + [{"role": "assistant", "content": assistant_msg}]
                        history_updated = True
                        print(f"[DEBUG] LLM 回复已添加到 history")
                except queue.Empty:
                    break

            # 返回当前状态 - 只有在 history 有变化时才更新 chatbot
            status_msg = "🎤 正在录音..."
            if state.error_message:
                status_msg = f"❌ {state.error_message}"
            elif state.current_transcript:
                status_msg = f"🎤 最新转写: {state.current_transcript}"
            
            if history_updated:
                return status_msg, history, state
            else:
                # 使用 gr.update() 跳过 chatbot 更新以避免闪烁
                return status_msg, gr.update(), state

        except Exception as e:
            print(f"[DEBUG] 流式音频处理错误: {e}")
            return f"❌ 音频处理错误: {e}", gr.update(), state

    def stop_and_process(
        self,
        history: list,
        state: SessionState | None,
    ) -> Generator[tuple[str, list, SessionState | None], None, None]:
        """停止录音并处理最后的转写结果"""
        print(f"[DEBUG] stop_and_process 被调用")
        
        if state is None:
            yield "没有活动的录音会话", history, None
            return

        try:
            # 停止会话
            state.stop_event.set()
            state.audio_queue.put("STOP_SESSION")  # 发送停止信号给 WebSocket worker
            state.llm_request_queue.put("STOP")
            state.is_active = False
            
            # 等待剩余的 LLM 响应
            import time
            time.sleep(0.5)
            
            # 收集剩余的 LLM 响应
            while True:
                try:
                    event_type, user_msg, assistant_msg = state.llm_response_queue.get_nowait()
                    if event_type == "response":
                        history = history + [{"role": "assistant", "content": assistant_msg}]
                except queue.Empty:
                    break
            
            yield "✅ 录音已停止", history, None

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] 处理错误: {e}")
            yield f"❌ 处理失败: {e}", history, None

    def clear_chat(self) -> tuple[list, str]:
        """清空对话"""
        self.messages = [
            {"role": "system", "content": self.llm_config.system_prompt}
        ]
        return [], ""

    def build_ui(self) -> gr.Blocks:
        """构建 Gradio UI"""
        with gr.Blocks(
            title="🍔 语音 + MCP 智能助手",
            theme=gr.themes.Soft(),
        ) as demo:
            gr.Markdown("""
            # 🍔 语音 + MCP 智能助手
            
            支持 **语音输入** 和 **文本输入**，可调用 MCP 工具完成任务。
            
            **使用方法**：点击麦克风开始录音，实时显示转写结果，说完后点击「停止并发送」。
            """)

            # 会话状态
            session_state = gr.State(value=None)

            with gr.Row():
                # 左侧控制面板
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 连接设置")
                    with gr.Group():
                        connect_btn = gr.Button("🔌 连接服务", variant="primary", size="lg")
                        disconnect_btn = gr.Button("断开连接", variant="secondary")
                        status_box = gr.Textbox(
                            label="连接状态",
                            value="未连接",
                            interactive=False,
                            lines=3,
                        )

                    gr.Markdown("### 🎤 语音输入（实时转写）")
                    with gr.Group():
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="点击麦克风开始录音（自动连接）",
                            streaming=True,
                        )
                        stop_btn = gr.Button("🛑 停止并发送", variant="primary")
                        voice_status = gr.Textbox(
                            label="实时转写",
                            value="点击麦克风开始录音...",
                            interactive=False,
                            lines=2,
                        )

                # 右侧聊天区域
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话",
                        height=450,
                    )

                    with gr.Row():
                        text_input = gr.Textbox(
                            label="文本输入",
                            placeholder="输入消息或使用语音...",
                            scale=4,
                            lines=1,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)

                    clear_btn = gr.Button("🗑️ 清空对话")

            # ========== 事件绑定 ==========

            # 连接/断开
            connect_btn.click(
                fn=self.connect,
                outputs=[status_box],
            )
            disconnect_btn.click(
                fn=self.disconnect,
                outputs=[status_box],
            )

            # 文本输入
            text_input.submit(
                fn=self.process_text,
                inputs=[text_input, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[text_input],
            )

            send_btn.click(
                fn=self.process_text,
                inputs=[text_input, chatbot],
                outputs=[chatbot],
            ).then(
                fn=lambda: "",
                outputs=[text_input],
            )

            # 流式语音输入 - 边录边转（只更新状态，不直接更新 chatbot）
            stream_event = audio_input.stream(
                fn=self.process_audio_stream,
                inputs=[audio_input, chatbot, session_state],
                outputs=[voice_status, chatbot, session_state],
                stream_every=0.5,  # 0.5秒检查一次
                time_limit=60,
                concurrency_limit=1,
            )

            # 停止录音并处理
            stop_btn.click(
                fn=self.stop_and_process,
                inputs=[chatbot, session_state],
                outputs=[voice_status, chatbot, session_state],
                cancels=[stream_event],
            )

            # 清空对话
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, text_input],
            )

        return demo

    def launch(self, **kwargs):
        """启动应用"""
        demo = self.build_ui()
        demo.queue()
        demo.launch(**kwargs)


def create_app(
    mcp_url: str,
    mcp_token: str | None = None,
    llm_model: str = "qwen3:8B",
    system_prompt: str | None = None,
) -> VoiceMCPGradioApp:
    """创建应用实例"""
    mcp_config = MCPConfig(mcp_url=mcp_url, mcp_token=mcp_token)
    llm_config = LLMConfig(
        model=llm_model,
        system_prompt=system_prompt or "你是一个智能助手，可以帮助用户完成各种任务。请用中文回答。",
    )

    return VoiceMCPGradioApp(
        mcp_config=mcp_config,
        llm_config=llm_config,
    )


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    app = create_app(
        mcp_url="https://mcp.mcd.cn/mcp-servers/mcd-mcp",
        llm_model="qwen3:8B",
        system_prompt="""你是麦当劳智能助手，可以帮助用户：
- 查询当前时间 (now-time-info)
- 查询活动日历 (campaign-calender)
- 查询可用优惠券 (available-coupons)
- 以及其他麦当劳相关服务

请用中文回答，简洁明了。当需要查询信息时，请调用相应的工具。""",
    )
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
