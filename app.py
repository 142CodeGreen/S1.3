import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index")

import torch
import os
import gradio as gr
import logging
from nemoguardrails import LLMRails, RailsConfig
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Ensure GPU usage
#if torch.cuda.is_available():
#    logger.info("GPU is available and will be used.")
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#else:
#    logger.warning("GPU not detected or not configured correctly. Falling back to CPU.")

# Initialize NeMo Guardrails here, outside of load_documents
config = RailsConfig.from_path("./Config")
rails = LLMRails(config)

index = None
query_engine = None

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    global index, query_engine
    if index is not None:
        return "Documents already loaded."

    try:
        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        #vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",
        #    gpu_id=0
        #)

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

        # Test query_engine synchronously to avoid mixing async with sync
        def test_query_engine():
            if query_engine:
                try:
                    response = query_engine.query("Test Query")
                    logger.info(f"Test query response: {response.response}")
                except Exception as e:
                    logger.error(f"Query engine failed with error: {str(e)}")
            else:
                logger.error("Query engine is not initialized")
        
        # Run the test in the same synchronous context
        test_query_engine()

        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

async def chat(message, history):
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.", None)]
    
    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": message}]
        )
        return history + [(message, response)]
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

async def query_guardrails(prompt):
    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}]
        )
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def stream_response(message, history):
    global query_engine
    if query_engine is None:
        yield history + [("Please upload a file first.", None)]
        return

    try:
        # Use asyncio.run to execute the async function in a synchronous context
        response = asyncio.run(query_guardrails(message))
        for text in response.split():  # Adjust streaming as needed
            history.append((message, text))
            yield history
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot with Guardrails")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load Documents")

    load_output = gr.Textbox(label="Load Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question", interactive=True)
    clear = gr.Button("Clear")

    # Synchronous function for loading documents
    load_btn.click(fn=load_documents, inputs=[file_input], outputs=[load_output])

    # Asynchronous function for streaming chat response
    msg.submit(fn=stream_response, inputs=[msg, chatbot], outputs=[chatbot])

    # Clear button functionality
    clear.click(lambda: [], None, chatbot, queue=False)

# Run the app
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
