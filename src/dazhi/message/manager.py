import asyncio
import json
import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (
    RealtimeConversationItemFunctionCallOutput,
    RealtimeResponseCreateParams,
)

from .history_item import (
    AssistantMessageHistoryItem,
    MCPCallMessageHistoryItem,
    MessageHistoryItem,
    SystemMessageHistoryItem,
    ToolResultMessageHistoryItem,
    ToolsCallMessageHistoryItem,
    UserMessageHistoryItem,
)

logger = logging.getLogger(__name__)


class MessageManager:

    def __init__(self, tool_executors: Optional[Dict[str, Callable]] = None):
        self.message_histories: OrderedDict[str, MessageHistoryItem] = OrderedDict()
        self.tool_executors = tool_executors if tool_executors is not None else {}

    async def handle_event(
        self, event: Any, connection: AsyncRealtimeConnection = None
    ) -> None:
        event_type: str = event.type
        names = event_type.split(".")
        if names[0] == "session":
            # 建立连接
            message_id = "session_init"
            system_item = self.update_message(message_id, "system")
            if names[1] == "created":
                system_item.connect()
            elif names[1] == "updated":
                system_item.update(event.session)
            else:
                logger.info(f"Unhandled session event: {event}")
        elif names[0] == "input_audio_buffer":  # VAD 事件
            item_id = event.item_id
            message_item = self.update_message(item_id, "user")
            self.route_vad(names, message_item, event)
        elif names[0] == "response":
            if names[1] == "created" or names[1] == "done":
                return  # 忽略整体 response 事件
            if names[1] == "output_item":
                return  # 忽略整体 output_item 事件
            if names[1] == "content_part":
                return  # 忽略整体 content_part 事件
            if names[1] == "output_text":
                message_item = self.update_message(event.item_id, "assistant")
                self.route_response_text(names, message_item, event)
            elif names[1] == "function_call_arguments":
                message_item = self.update_message(event.item_id, "tools_call")
                if names[2] == "done":
                    tool_result = await self.route_function_call_arguments(
                        names, message_item, event
                    )
                    self.update_message(
                        event.item_id + "_result", "tool_result"
                    ).update(
                        message_item.tool_name, message_item.tool_arguments, tool_result
                    )
                    try:
                        await self.send_tool_result(
                            connection, event.call_id, tool_result
                        )
                    except Exception as e:
                        logger.error(f"Error sending tool result: {e}")
            elif names[1] == "mcp_call_arguments":
                logger.info(f"MCP Call Arguments Event: {event_type}")
            elif names[1] == "mcp_call":
                logger.info(f"MCP Call Event: {event_type}")
            else:
                logger.info(f"Unhandled response event: {event_type}")
        elif names[0] == "conversation":
            await self.route_conversation(names, event, connection)
        elif names[0] == "rate_limits":
            logger.info(event_type)
        elif names[0] == "error":
            logger.error(f"Error event received: {event}")
        elif names[0] == "mcp_list_tools":
            logger.info(f"MCP Tools Event: {event}")
        else:
            logger.info(f"Unhandled event type: {event_type}")
        return

    def render_gradio_history(self) -> List[Dict[str, str]]:
        """渲染消息历史以适应 Gradio 聊天界面"""
        rendered_history = []
        for item in self.message_histories.values():
            content = item.render()
            if content:
                rendered_history.append({"role": item.role, "content": content})
        return rendered_history

    def update_message(
        self,
        message_id: str,
        role: Optional[
            Literal[
                "system",
                "user",
                "assistant",
                "tools_call",
                "tool_result",
                "mcp_call",
                "mcp_result",
            ]
        ] = None,
    ) -> MessageHistoryItem:
        """如消息存在，找到对应消息并返回，否则创建新消息"""
        if message_id in self.message_histories:
            return self.message_histories[message_id]
        if role == "system":
            item = SystemMessageHistoryItem()
        elif role == "user":
            item = UserMessageHistoryItem()
        elif role == "assistant":
            item = AssistantMessageHistoryItem()
        elif role == "tools_call":
            item = ToolsCallMessageHistoryItem()
        elif role == "tool_result":
            item = ToolResultMessageHistoryItem()
        elif role == "mcp_call":
            item = MCPCallMessageHistoryItem()
        else:
            raise ValueError(f"Unknown role: {role}")
        self.message_histories[message_id] = item
        return item

    def route_vad(self, names: List[str], message_item: UserMessageHistoryItem, event):
        if names[1] == "speech_started":
            message_item.start_time = event.audio_start_ms / 1000
        elif names[1] == "speech_stopped":
            message_item.end_time = event.audio_end_ms / 1000
        elif names[1] == "committed":
            pass
        else:
            logger.info(f"Unhandled input_audio_buffer event: {event.type}")

    async def route_conversation(
        self, names: List[str], event, connection: AsyncRealtimeConnection = None
    ):
        if names[1] == "item":
            if names[2] == "added":
                pass
            elif names[2] == "done":
                item_type = event.item.type
                if item_type == "mcp_call":
                    message_item: MCPCallMessageHistoryItem = self.update_message(
                        event.item.id, "mcp_call"
                    )
                    message_item.update_arguments(
                        f"{event.item.server_label}.{event.item.name}",
                        event.item.arguments,
                    )
                    message_item.update_result(event.item.output)
                    await self.send_response_create(connection)
                elif item_type == "mcp_list_tools":
                    logger.info(f"MCP List Tools done event")
                elif item_type == "message":
                    if event.item.role == "user":
                        logger.info(f"User message item done event")
                    elif event.item.role == "assistant":
                        logger.info(f"Assistant message item done event")
                else:
                    logger.info(
                        f"Unhandled conversation.item.done event: {event.type} with item type {item_type}"
                    )
            elif names[2] == "input_audio_transcription":
                item_id = event.item_id
                message_item = self.update_message(item_id)
                if names[3] == "delta":
                    message_item.update_stream(event.delta)
                elif names[3] == "completed":
                    message_item.finish()
                else:
                    logger.info(
                        f"Unhandled conversation.input_audio_transcription event: {event}"
                    )
            else:
                logger.info(f"Unhandled conversation.item event: {event}")
        else:
            logger.info(f"Unhandled conversation event: {event}")

    def route_response_text(
        self, names: List[str], message_item: UserMessageHistoryItem, event
    ):
        if names[2] == "delta":
            message_item.update_stream(event.delta)
        elif names[2] == "done":
            message_item.finish()
        else:
            logger.info(f"Unhandled response.output_text event: {event}")

    async def route_function_call_arguments(
        self, names: List[str], message_item: ToolsCallMessageHistoryItem, event
    ) -> str:
        message_item.update(event.name, event.arguments)
        try:
            arguments = json.loads(event.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding tool arguments: {e}")
            tool_result = f"Error decoding tool arguments: {e}"
            return tool_result
        tool_result = await self.use_tool(event.name, arguments)
        return tool_result

    async def use_tool(self, name: str, arguments: Dict) -> str:
        """调用工具并返回结果"""
        tool_executor = self.tool_executors.get(name)
        if not tool_executor:
            return f"Tool {name} not found"
        try:
            if asyncio.iscoroutinefunction(tool_executor):
                result = await tool_executor(arguments)
            else:
                result = tool_executor(arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {e}"

    async def send_tool_result(
        self, connection: AsyncRealtimeConnection, call_id: str, output: str
    ):
        await connection.conversation.item.create(
            item=RealtimeConversationItemFunctionCallOutput(
                call_id=call_id, output=output, type="function_call_output"
            )
        )
        await connection.response.create(response=RealtimeResponseCreateParams())

    async def send_response_create(self, connection: AsyncRealtimeConnection):
        await connection.response.create(response=RealtimeResponseCreateParams())
