import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index")

import torch
import os
import logging

import gradio as gr
import asyncio
from Config.actions import load_documents, init
from nemoguardrails import LLMRails, RailsConfig

if torch.cuda.is_available():
    print(f"CUDA is available, GPU being used: {torch.cuda.get_device_name(0)}")
    print(f"Total CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
else:
    print("CUDA is not available. Make sure you have a GPU with CUDA installed.")

# Initialize NeMo Guardrails
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)
init(rails)  # Initialize your registered actions

async def stream_response(message, history):
    try:
        # Here we assume that NeMo Guardrails handles the generation
        response = await rails.generate_async(
            messages=[{"role": "user", "content": message}]
        )
        partial_response = ""
        for token in response.split():  # Adjust streaming as needed
            partial_response += token + " "
            yield history + [(message, partial_response)]
    except Exception as e:
        logging.error(f"An error occurred in stream_response: {e}")
        yield history + [(message, f"Error processing query: {str(e)}")]


with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot with NeMo Guardrails")
    
    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load Documents")

    load_output = gr.Textbox(label="Load Status")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question after document is loaded", interactive=True)
    clear = gr.Button("Clear")

    # Event handlers
    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch(share=True, debug=True)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True,debug=True)
