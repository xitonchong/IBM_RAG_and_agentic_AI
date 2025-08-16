''' MOdule for p[rocessing Linkedin profile data'''

from icecream import ic
import json
import logging 
from typing import Dict, List, Any, Optional
import config 

from llama_index.core import Document, VectorStoreIndex 
from llama_index.core.node_parser import SentenceSplitter 
from modules.llm_interface import create_local_embedding


logger = logging.getLogger(__name__) 

def split_profile_data(profile_data: Dict[str, Any]) -> List: 
    ''' splits the linkedin profile JSON data into nodes. 
        args: 
            profile_data:linkedin profile data dictionary 
        returns: 
            list of document nodes. 
    '''
    try: 
        # convert the profile data to a JSON string 
        profile_json = json.dumps(profile_data) 

        #create a document object from the JSON string 
        document = Document(text=profile_json)

        # split the document into nods using sentence splitter 
        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE) 
        nodes = splitter.get_nodes_from_documents([document])

        logger.info(f"created {len(nodes)} nodes from profile data")
        return nodes 

    except Exception as e: 
        logger.error(f"Error in split_profile_data: {e}")
        return []
    

def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]: 
    ''' Stores the document chunks (nodes) into vector database. 

    Args: 
        nodes: List of document nodes to be indexed. 

    returns:
        VectorStoreIndex or NOne if indexing fails.
    
    '''

    try: 
        # embedding model 
        embedding_model = create_local_embedding()
        # embedding_model = create_watson_embedding()

        # create a vectorStoreIndex from the nodes 
        index = VectorStoreIndex(
            nodes=nodes, 
            embed_model=embedding_model, 
            show_progress=True, 
        )

        logger.info(f"Vector databse created successfully")
        return index 
    except Exception as e: 
        logger.error(f"error in create_vector_database: {e}")
        return None
    

def verify_embeddings(index: VectorStoreIndex) -> bool:
    ''' Verify that all nodes have been properly embedded
        Args: 
            index: VectorStoreIndex to verify 
        Returns:
            True if all embedings are valid, False otherwise 
    '''
    try:
        vector_store = index._storage_context.vector_store 
        node_ids = list(index.index_struct.nodes_dict.keys())
        missing_embeddings = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id) 
            if embedding is None: 
                logger.warning(f"Node ID {node_id} has a None embedding.")
                missing_embeddings = True 
            
        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False 
        else: 
            logger.info("All node embeddings are valid")
        return True 
    except Exception as e: 
        logger.error(f"Error in verify_embeddings: {e}")    
        return False 
