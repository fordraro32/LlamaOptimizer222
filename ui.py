import gradio as gr
from text_generation import generate_text

def create_ui(model, tokenizer, conversation_manager):
    def chat(message, history):
        response = generate_text(model, tokenizer, message, conversation_manager)
        history.append((message, response))
        return "", history

    with gr.Blocks() as iface:
        gr.Markdown("# Hermes-3-Llama Optimizer")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    return iface
