import gradio as gr

def answer(message, history):
    return "今天天气真好"

demo = gr.ChatInterface(
    answer,
    title="Stub Chatbot",
    description="无论你问什么，我都会说今天天气真好。",
    textbox=gr.MultimodalTextbox(
        file_types=[".pdf", ".txt"],
        placeholder="随便输入点什么或上传文件"
    ),
    multimodal=True,
    api_name="chat",
)

demo.launch()
