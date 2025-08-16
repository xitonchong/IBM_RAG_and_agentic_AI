import logging 
import config 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def create_llm(
    temperature = config.TEMPERATURE, 
		decoding_method: str = "sample"
	):
  if config.LLM_MODEL_ID == "gemini-1.5-flash":
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_ID, temperature=temperature)
  else: 
     raise NotImplementedError(f"LLM {config.LLM_MODEL_ID} is not implemented!")
  return llm


def change_llm_model(new_model_id: str) -> None: 
    ''' change the LLM model to use'''
    global config 
    config.LLM_MODEL_ID = new_model_id 
    logger.info(f"Changed LLM model to: {new_model_id}")
    


def create_local_embedding():
    """Creates a local HuggingFace Embedding model for vector representation.
    
    Returns:
        HuggingFaceEmbeddings model.
    """
    local_embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID)
    logger.info(f"Created local Embedding model: {config.EMBEDDING_MODEL_ID}")
    return local_embedding