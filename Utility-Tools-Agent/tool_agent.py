import sys 
import logging 


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout), 
        logging.FileHandler('app.log')
    ]
)
import numpy as np
import pandas as pd 
import matplotlib 
import seaborn 
import sklearn 
import langchain 
import openai 
import langchain_openai 
from langchain_ollama import ChatOllama
from df_tools import * 
from model_tools import * 

import glob 
import os 
from typing import List, Optional 

from langchain.agents import tool, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model 
from langchain.agents import AgentExecutor


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a data science assistant. Use the available tools to analyze CSV files. "
     "Your job is to determine whether each dataset is for classification or regression, based on its structure."),
    
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # Required for tool-calling agents
])
llm = init_chat_model("gpt-oss:20b", model_provider="ollama", streaming=False)
# llm = ChatOllama(model="gpt-oss:20b")
tools=[list_csv_files, preload_datasets, get_dataset_summaries, 
       call_dataframe_method, evaluate_classification_dataset, 
       evaluate_regression_dataset]

# constrcut the tool calling agent 
agent = create_openai_tools_agent(llm, tools, prompt )


response = agent.invoke({
    "input": "Can you tell me about the dataset?", 
    "intermediate_steps": []
})

# get the first ToolAgentAction from the list 
action = response[0]

#print the key details 
print("Agent decide to call a tool: ") 
print("tool name: ", action.tool) 
print("too input: ", action.tool_input)
print("log: \n", action.log.strip())


'''
when the agent was called with the input "Can you tell me about the dataset?"
it responded with a tool action: it chose to invoke list_csv_files without any 
arguments. It didn't try to load or analyze the dataset 

ReAct style agents follow a step by step reasoning loop. react stands for 
reasoning and acting: the agent thinks about what to do next, 
takes one action (like calling a tool), 
then waits for the result before continuing. 
This is why the agent's first instinct is to gather context—by 
listing the available CSV files—before attempting anything 
more complex. This isn’t a failure
; it’s how the agent is designed to operate—reasoning one 
step at a time based on feedback.
'''

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)
agent_executor.agent.stream_runnable = False

while True:
    user_input=input(" You:")
    if user_input.strip().lower() in ['exit','quit']:
        print("see ya later")
        break
        
    result=agent_executor.invoke({"input":user_input})
    print(f"my Agent: {result['output']}")