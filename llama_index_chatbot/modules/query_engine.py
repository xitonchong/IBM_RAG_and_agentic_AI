"""Module for querying indexed Linkedin profile data."""

from langchain_google_genai import ChatGoogleGenerativeAI

import logging 
from typing import Any, Dict, Optional 

from llama_index.core import VectorStoreIndex, PromptTemplate 


from modules.llm_interface import create_local_embedding 
import config 


logger = logging.getLogger(__name__)

def generate_initial_facts(index: VectorStoreIndex) -> str: 
    ''' Generates interesting facts about the person's career or eduction. 
    
    Args: 
        index: vectorStoreIndex containing the Linkedin profile data. 
    
    Returns: 
        string containing interesting facts about the person
    '''
    try: 
        # Create LLM for generating facts 
        llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_ID, temperature=config.TEMPERATURE)

        # Create prompt template 
        facts_prompt = PromptTemplate(template=config.INITIAL_FACTS_TEMPLATE)

        # create query engine 
        query_engine  = index.as_query_engine(
            streaming=False, 
            similarity_top_k=config.SIMILARITY_TOP_K, 
            llm=llm, 
            text_qa_template=facts_prompt 
        )

        # execute the query 
        query = "Provide three interesting facts about this person's career or eduction."
        response = query_engine.query(query)

        # return the facts 
        return response.response
    except Exception as e: 
        logger.error(f"Error in generate_initial_facts: {e}")
        return "Failed to generate initial facts."
    

def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any: 
    try: 
        # create llm 
        llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_ID, temperature=config.TEMPERATURE)


        # create prompt template 
        question_prompt = PromptTemplate(template=config.USER_QUESTION_TEMPLATE)


        # Retrieve relevant nodes 
        base_retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        source_nodes = base_retriever.retrieve(user_query) 

        # build context string 
        context_str = "\n\n".join([node.node.get_text() for node in source_nodes])

        # create query engine 
        query_engine = index.as_query_engine(
            streaming=False, 
            similarity_top_k=config.SIMILARITY_TOP_K, 
            llm=llm, 
            text_qa_template = question_prompt, 
            context_str = context_str
        )

        # execute the query 
        answer = query_engine.query(user_query) 
        return answer

    except Exception as e: 
        logger.error(f"Error in answer_user_query: {e}")
        return "Failed to get an answer"