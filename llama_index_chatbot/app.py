''' gradio web interface for the icebreaker bot '''


import os 
import sys 
import uuid 
import logging 
from modules.llm_interface import change_llm_model
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database
from modules.data_processing import verify_embeddings
from modules.query_engine import generate_initial_facts, answer_user_query
from icecream import ic
import config 
import gradio as gr
from dotenv import load_dotenv


load_dotenv() 


logging.basicConfig(
	level=logging.INFO, 
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(stream=sys.stdout),
		logging.FileHandler('app.log')
	]
)


logger = logging.getLogger(__name__)

# dictionary to stroe active conversations 
active_indices = {} 

def process_profile(linkedin_url, api_key, use_mock, selected_model): 
  '''Process a LinkedIn Profile and generate initial facts. 
  Args: 
  linkedin_url: linkedin profile URL to process. 
  api_key: ProxyCurl API key. 
  use_mock: Whether to use mock data. 
  selecdted_model: LLM model to use. 
  
  returns: 
  initial facts about the profile and a session ID for this conversation. 
  '''
  try: 
      # Change LLM model if needed 
    if selected_model != config.LLM_MODEL_ID: 
      change_llm_model(selected_model) 
      
    # Use a deafult URL for mock data if none provided 
    if use_mock and not linkedin_url:
      linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"
          
    

    # extract profile data 
    profile_data = extract_linkedin_profile(
        linkedin_url, 
        api_key if not use_mock else None,
        mock=use_mock
    )

    if not profile_data: 
      return f""". Failed to retrieve profile data. Please check the URL or API key. """, None

    # split data into nodes 
    nodes = split_profile_data(profile_data) 

    if not nodes: 
      return f" Failed to process profile data into nodes. ", None 
    

    # create vector databse 
    index = create_vector_database(nodes) 
    
    if not index:
      return "failed to create vector database. ", None 
  
    # Verify Embeddings 
    if not verify_embeddings(index): 
      logger.warning("Some embedding may be missing or invalid")

    # Generate initial facts 
    facts = generate_initial_facts(index) 
    ic(facts)

    # Generate a unique session ID 
    session_id = str(uuid.uuid4())

    # store the index for this session 
    active_indices[session_id] = index 

    # return the facts and session ID 
    return f"Profile processed successfully!\n\n Here are 3 interesting facts about this person: \n\n {facts}", session_id


  except Exception as e: 
    logger.error(f"error in processing process_profile: {e}")
    print(e)
    return f"Error: {str(e)}", None



def chat_with_profile(session_id, user_query, chat_history): 
  '''Chat with a processed LinkedIn profile

    Args: 
      session_id: session id for this conversion. 
      user_query: User's question.
      chat_history:  Chat history

    Returns: 
      updated chat history
  ''' 
  if not session_id: 
    return chat_history + [[user_query, "No profile loaded. Please process a linkedin profile first"]]
  
  if session_id not in active_indices: 
    return chat_history + [[user_query, "Session expired. Please process the linkedin profile again"]]
  
  if not user_query.strip(): 
    return chat_history 
  
  try: 
    # get the index for this session 
    index = active_indices[session_id]

    # answer the user's query 
    response = answer_user_query(index, user_query)

    # update chat history 
    return chat_history + [[user_query, response.response]]

  except Exception as e: 
    logger.error(f"Error in chat_with_profile: {e}")
    return chat_history + [[user_query, f"Error: {str(e)}"]]





def create_gradio_interface():
    """Create the Gradio interface for the Icebreaker Bot."""
    # Define available LLM models
    available_models = [
      config.LLM_MODEL_ID
    ]
    
    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles")
        
        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL",
                        placeholder="https://www.linkedin.com/in/username/"
                    )
                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (Leave empty to use mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password",
                        value=config.PROXYCURL_API_KEY
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Select LLM Model",
                        value=config.LLM_MODEL_ID
                    )
                    process_btn = gr.Button("Process Profile")
                
                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)
            
            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id]
            )
        
        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")
            
            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile",
                placeholder="What is this person's current job title?"
            )
            
            chat_btn = gr.Button("Send")
            
            chat_btn.click(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot]
            )
            
            chat_input.submit(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot]
            )
    return demo

if __name__ == "__main__":
    # demo = create_gradio_interface()
    # # Launch the Gradio interface
    # # You can customize these parameters:
    # # - share=True creates a public link you can share with others
    # # - server_name and server_port set where the app runs
    # demo.launch(
    #     server_name="127.0.0.1",  
    #     server_port=5000,
    #     share=True  # Set to False if you don't want to create a public link
    # )
    chat_history=[]
    interesting_facts, session_id = process_profile('', None, use_mock=True, selected_model=config.LLM_MODEL_ID)
    while True:
      user_input = input('type query\n')
      if user_input in ['quit', 'break']:
        break

      chat_history = chat_with_profile(session_id, user_input, chat_history=chat_history)
      ic(chat_history[-1:])