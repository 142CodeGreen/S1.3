from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails import LLMRails, RailsConfig
#from nemoguardrails.kb.kb import KnowledgeBase

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
query_engine = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        kb_dir = "./Config/kb"
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
            shutil.copy2(file_path, kb_dir)

        if not documents:
            return "No documents found in the selected files."

        # Initialize Milvus Vector Store
          # use GPU for Milvus workload
        #vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",
        #    gpu_id=0
        #)  
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        
        # Vectorize documents
        embeddings = [Settings.embed_model.get_text_embedding(doc.text) for doc in documents]
        
        # Insert vectors into Milvus
        for doc, embedding in zip(documents, embeddings):
            vector_store.insert([embedding], doc.metadata)

        # Create index if it doesn't exist
        if not index:
            index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=vector_store))
            query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

        return f"Successfully loaded documents into Milvus."
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return f"Error loading documents: {str(e)}"
        
   
def template(question, context):
    return f"""Answer user questions based on loaded documents. 
    
    {context}
    
    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

@action(is_system_action=True)
async def rag(context: dict, llm, kb) -> ActionResult:
    global query_engine
    try:
        message = context.get('last_user_message', '')
        if not message:
            return ActionResult(return_value="No user query provided.", context_updates={})
        
        # Vectorize query
        query_embedding = Settings.embed_model.get_text_embedding(message)
        
        # Search for similar vectors in Milvus
        results = vector_store.search(query_embedding, limit=20)  # Assuming similarity_top_k=20
        relevant_chunks = [vector_store.get_document(result.id) for result in results]
        
        context_text = "\n".join([chunk.text for chunk in relevant_chunks])

        prompt = template(message, context_text)
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
