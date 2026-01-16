import json
import os
from typing import Any, Dict

import dotenv
import httpx
import openai
from openai.types.chat.chat_completion_function_tool import ChatCompletionFunctionTool
from openai.types.shared.function_definition import FunctionDefinition

dotenv.load_dotenv()
base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:10002/v1")
api_key = os.getenv("TEST_API_KEY", "dummy_api_key")
print(f"Using base_url: {base_url}")

client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=httpx.Client(verify=False),  # 🔴 关键：关闭证书校验
)

tools = ChatCompletionFunctionTool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="获取一个地点的天气，用户应该先提供一个地点。",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名，例如：北京，上海，广州",
                }
            },
            "required": ["location"],
        },
    ),
)


def get_weather(location: str) -> Dict[str, Any]:
    # 这里换成你自己的真实逻辑
    return {"location": location, "temperature_c": 114514}


def send_messages(messages: list[dict]) -> dict:
    response = client.chat.completions.create(
        model="qwen3:8B",
        messages=messages,
        tools=[tools.model_dump()],
        stream=True,
    )
    # print(response.choices[0].message)

    content_chunks = []
    tool_calls = []
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # print(chunk.model_dump())
        if delta is None:
            continue
        if delta.content:
            content_chunks.append(delta.content)
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                while len(tool_calls) <= tool_call.index:
                    tool_calls.append(
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": None, "arguments": ""},
                        }
                    )
                target = tool_calls[tool_call.index]
                if tool_call.id:
                    target["id"] = tool_call.id
                if tool_call.function and tool_call.function.name:
                    target["function"]["name"] = tool_call.function.name
                if tool_call.function and tool_call.function.arguments:
                    target["function"]["arguments"] += tool_call.function.arguments
    if content_chunks:
        print("")
    message = {"role": "assistant", "content": "".join(content_chunks)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


system_prompt = "你必须用中文回答我"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "今天金华的天气怎么样？ /no_think"},
]
print(f"User>\t {messages}")
print("Model>\t ", end="", flush=True)
message = send_messages(messages)


messages.append(message)

if "tool_calls" in message:
    tool = message["tool_calls"][0]
    args = json.loads(tool["function"]["arguments"])

    if tool["function"]["name"] == "get_weather":
        result = get_weather(**args)
    else:
        result = {"error": f"Unknown tool: {tool['function']['name']}"}

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool["id"],
            "content": json.dumps(result, ensure_ascii=False),
        }
    )

    print("Model>\t ", end="", flush=True)
    message = send_messages(messages)
    print(f"Model>\t {message['content']}")
