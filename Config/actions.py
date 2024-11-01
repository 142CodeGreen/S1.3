from nemoguardrails import LLMRails, RailsConfig, Action, ActionResult
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
import os
import asyncio
import shutil

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    if not file_objs:
        return "No files selected."
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)
    
    file_paths = get_files_from_input(file_objs)
    documents = []
    for file_path in file_paths:
        # Copy file to kb directory
        shutil.copy2(file_path, kb_dir)
        # Load document for indexing if you're still managing an index
        documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
    
    if not documents:
        return f"No documents found in the selected files."

    # Here you might index if you're not relying entirely on NeMo Guardrails for retrieval
    # vector_store = MilvusVectorStore(...)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

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
    try:
        message = context.get('last_user_message', '')
        if not message:
            return ActionResult(return_value="No user query provided.", context_updates={})
        
        # Setup Milvus vector store
        vector_store = MilvusVectorStore(
            host="127.0.0.1",
            port=19530,
            dim=1024,
            collection_name="your_collection_name",
            gpu_id=0
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from documents in kb folder
        documents = SimpleDirectoryReader(input_dir="./Config/kb").load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        
        # Query the index for relevant information
        response = await query_engine.aquery(message)
        relevant_chunks = "\n".join([node.text for node in response.source_nodes])

        # Generate the answer using the LLM with the template
        prompt = template(message, relevant_chunks)
        answer = await llm.generate_async(prompt)
        
        context_updates = {
            'last_bot_message': answer.text,
            '_last_bot_prompt': prompt
        }
        
        return ActionResult(return_value=answer.text, context_updates=context_updates)
    except Exception as e:
        return ActionResult(return_value=f"Error processing query: {str(e)}", context_updates={})

def init(app: LLMRails):
    app.register_action(rag, "rag")
