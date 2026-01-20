import random

import gradio as gr
from gradio import ChatMessage

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        if not message:
            return "", chat_history
        bot_message = random.choice(
            ["How are you?", "Today is a great day", "I'm very hungry"]
        )
        chat_history.append(ChatMessage(role="user", content=message))
        chat_history.append(ChatMessage(role="assistant", content=bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
