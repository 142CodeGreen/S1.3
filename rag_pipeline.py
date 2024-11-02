from nemoguardrails import LLMRails, ActionResult
from llama_index.core import Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from ..doc_loader import load_documents
import logging

logger = logging.getLogger(__name__)

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

def template(question, context):
    return f"""Answer user questions based on loaded documents. 

    {context}

    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

async def rag(context: dict, llm, kb, query_engine):
    message = context.get('last_user_message', '')
    if not message:
        return ActionResult(return_value="No query provided", context_updates={})
    
    try:
        response = await query_engine.aquery(message)  # Assuming there's an async query method
        relevant_chunks = response.response

        prompt = template(message, relevant_chunks)
        answer = await llm.apredict(prompt)  # Assuming there's an async predict method
        
        return ActionResult(return_value=answer.text, context_updates={
            'last_bot_message': answer.text,
            '_last_bot_prompt': prompt
        })
    except Exception as e:
        logger.error(f"Error in RAG process: {str(e)}")
        return ActionResult(return_value="An error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    app.register_action(rag, "rag")

async def initialize_rag(file_objs):
    index = await load_documents(file_objs)  # Assuming document loading could be async
    query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
    rails = LLMRails(config_path="./Config")
    init(rails)  # Initialize the guardrails with the RAG action
    return rails, query_engine
