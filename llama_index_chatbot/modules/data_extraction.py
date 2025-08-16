import requests
import time 
import logging
from typing import Dict, Optional, Any
import config 

logger = logging.getLogger(__name__)

def extract_linkedin_profile(
        linkedin_url: str, 
        api_key: Optional[str] = None, 
        mock: bool = False) -> Dict[str, Any]: 
    start_time = time.time()

    try: 
        if mock: 
            logger.info("Using mock data from a premade JSON file...")
            mock_url = config.MOCK_DATA_URL 
            response = requests.get(mock_url, timeout=30)
        else: 
            # ensure APIkey is provided when mock is False 
            if not api_key: 
                raise ValueError("ProxyCurl API key is required when mock is set to False")
            logger.info("Starting to extract the Linkedin Profile...")

            raise NotImplementedError("CurlProxy became obsolete")

        # check if response is successful 
        if response.status_code == 200:
            try: 
                data = response.json() 
                
                # clean the data, remove empty values and unwanted fields
                data = { 
                    k: v
                    for k, v in data.items() 
                    if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
                }

                # Remove profile picture URLs from groups to clean the data 
                if data.get("groups"): 
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)

                return data 
            except ValueError as e: 
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response content: {response.text[:200]}...")
                return {}
        else: 
            logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            logger.error(f"Response {response.text}")
            return {}
    except Exception as e: 
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}