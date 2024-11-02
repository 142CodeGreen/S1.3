import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores import MilvusVectorStore

def load_documents(file_objs):
    """
    Load documents from provided file objects and index them into Milvus.
    
    Parameters:
    - file_objs: A list of file objects to load documents from.
    
    Returns:
    - VectorStoreIndex: An index object that can be used for querying.
    """
    kb_dir = os.path.join('Config', 'kb')
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)

    documents = []
    for file_obj in file_objs:
        documents.extend(SimpleDirectoryReader(input_files=[file_obj.name]).load_data())

    vector_store = MilvusVectorStore(uri="milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    return index
