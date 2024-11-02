import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index")

import torch
import os
import logging

import gradio as gr
import asyncio
from rag_pipeline import initialize_rag

if torch.cuda.is_available():
    print(f"CUDA is available, GPU being used: {torch.cuda.get_device_name(0)}")
    print(f"Total CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
else:
    print("CUDA is not available. Make sure you have a GPU with CUDA installed.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rails, query_engine = None, None

async def stream_response(message, history):
    """
    Stream the response from the RAG system for the given message.

    Parameters:
    - message: The user's query.
    - history: The chat history to append the new message to.

    Yields:
    - List: Updated chat history with streaming responses.
    """
    if query_engine is None:
        yield history + [("Please upload a file first.", None)]
        return
    
    user_message = {"role": "user", "content": message}
    result = await rails.generate_async(messages=[user_message])
    
    for chunk in result:
        history.append((message, chunk)) 
        yield history
    history.append((None, "End of response.")) 
    yield history

async def load_and_setup(file_objs):
    """
    Load documents and set up the RAG system.

    Parameters:
    - file_objs: List of file objects to load into the system.

    Returns:
    - str: Status message indicating success or error.
    """
    global rails, query_engine
    try:
        rails, query_engine = await initialize_rag(file_objs)
        return "Documents loaded and RAG system initialized."
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return f"Error: {str(e)}"

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for PDF Files")
    file_input = gr.File(label="Select files to upload", file_count="multiple")
    load_btn = gr.Button("Load PDF Documents only")
    load_output = gr.Textbox(label="Load Status")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear")

    # Button click event to load documents
    load_btn.click(load_and_setup, inputs=[file_input], outputs=[load_output])
    
    # Textbox submission event for querying the chatbot
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    
    # Button click event to clear the chat
    clear.click(lambda: None, None, chatbot, queue=False)

async def main():
    # Run the Gradio interface
    await demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    asyncio.run(main())
