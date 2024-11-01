# config/rag.py

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from nemoguardrails import LLMRails, RailsConfig

# Initialize your LLM and Embedding Settings
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Vector Store Setup (if still needed)
def setup_vector_store():
    #vector_store = MilvusVectorStore(
    #    host="127.0.0.1",
    #    port=19530,
    #    dim=1024,
    #    collection_name="your_collection_name",
    #    gpu_id=0
    #)
    vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
    return vector_store

def load_documents(file_objs):
    if not file_objs:
        return "No files selected."
    kb_dir = "./Config/kb"  # Create the 'kb' directory if it doesn't exist
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)
    file_paths = get_files_from_input(file_objs)
    documents = []
    for file_path in file_paths:
        documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
        shutil.copy2(file_path, kb_dir)
    if not documents:
        return f"No documents found in the selected files.", gr.update(interactive=False)
  

async def rag(context, llm, kb):
    # Custom RAG logic if needed, otherwise leave this empty or use NeMo Guardrails default behavior
    pass

# NeMo Guardrails setup
def setup_rails():
    config = RailsConfig.from_path("./Config")
    rails = LLMRails(config)
    return rails

# Setup your index and query engine logic here if you decide to use them outside NeMo Guardrails
