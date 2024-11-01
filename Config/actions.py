from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.kb.kb import KnowledgeBase

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
import os
import asyncio
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Global variable to hold our index
index = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)
    
    file_paths = get_files_from_input(file_objs)
    for file_path in file_paths:
        try:
            shutil.copy2(file_path, kb_dir)
            logger.info(f"File copied: {file_path}")
        except Exception as e:
            logger.error(f"Failed to copy {file_path}: {str(e)}")
    
    if not file_paths:
        return "No files selected."
    return f"Documents loaded into {kb_dir}."

def template(question, context):
    return f"""Answer user questions based on loaded documents. 
    
    {context}
    
    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

@Action(is_system_action=True)
async def rag(context: dict, llm, kb: KnowledgeBase) -> ActionResult:
    global index
    
    try:
        message = context.get('last_user_message', '')
        if not message:
            return ActionResult(return_value="No user query provided.", context_updates={})
        
        # Create the index if it hasn't been created yet
        if index is None:
            #vector_store = MilvusVectorStore(
            #    host="127.0.0.1",
            #    port=19530,
            #    dim=1024,
            #    collection_name="your_collection_name",
            #    gpu_id=0
            #)

            vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            documents = SimpleDirectoryReader(input_dir="./Config/kb").load_data()
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            logger.info("Index created.")
        
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        response = await query_engine.aquery(message)
        relevant_chunks = "\n".join([node.text for node in response.source_nodes])

        prompt = template(message, relevant_chunks)
        answer = await llm.generate_async(prompt)
        
        context_updates = {
            'last_bot_message': answer.text,
            '_last_bot_prompt': prompt
        }
        
        return ActionResult(return_value=answer.text, context_updates=context_updates)
    except Exception as e:
        logger.error(f"Error in RAG process: {str(e)}")
        return ActionResult(return_value=f"An error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    app.register_action(rag, "rag")
