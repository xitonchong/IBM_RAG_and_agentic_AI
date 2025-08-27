# %% 
import re
from pytube import YouTube
from langchain_core.tools import tool 
import yt_dlp 
from typing import List, Dict 
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage 
import json 
from langchain.chat_models import init_chat_model 
from langchain_ollama import ChatOllama 
from custom_tools import * 
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Dict

import warnings 
warnings.filterwarnings("ignore")

# suppress pytube errors 
import logging

# Configure a logger for the script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("output.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# suppress pytube errors
pytube_logger = logging.getLogger("pytube")
pytube_logger.setLevel(logging.ERROR)

# %% 



tools = [] 
tools.append(extract_video_id) 
tools.append(fetch_transcript)
tools.append(search_youtube) 
tools.append(get_full_metadata)
tools.append(get_thumbnails)


llm = init_chat_model(model="gpt-oss:20b", model_provider="ollama")
llm_with_tools = llm.bind_tools(tools) 


for tool in tools: 
    schema = { 
        "name": tool.name, 
        "description": tool.description, 
        "parameters": tool.args_schema.schema() if tool.args_schema else {}, 
        "return": tool.return_type if hasattr(tool, "return_type") else None
    }
    print(schema)

query = "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"
print(query)
messages = [HumanMessage(content=query)]

from icecream import ic

# langchain tool binding process 
response_1= llm_with_tools.invoke(messages) 
ic(response_1)


messages.append(response_1) 

# create a tool mapping dictionary 
tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "get_trending_videos": get_trending_videos,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}

tool_calls_1 = response_1.tool_calls 
print(tool_calls_1) 

my_tool = tool_mapping[tool_calls_1[0]['name']]
video_id = my_tool.invoke(tool_calls_1[0]['args'])

print(video_id) 

# %% 
# adding the tool's output to your conversation history. you'll create 
# a toolMessage that contains: 
# 1. the result from executing the tool (the extracted video ID) 
# 2. the original tool call ID to linl this response back to specific request 


messages.append(ToolMessage(content=video_id, tool_call_id = tool_calls_1[0]['id']))

response_2 = llm_with_tools.invoke(messages) 
ic(response_2) 

# messages.append(response_2) 
# tool_calls_2 = response_2.tool_calls 
# ic(tool_calls_2) 

# fetch_transcript_tool_output = tool_mapping[tool_calls_2[0]['name']].invoke(tool_calls_2[0]['args'])
# ic(fetch_transcript_tool_output)

# %% 

def execute_tool(tool_call): 
    ''' executing single tool call and return ToolMessage'''
    try: 
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(
            content=str(result), 
            tool_call_id=tool_call["id"]
        )
    except Exception as e: 
        return ToolMessage(
            content=f"Error: {str(e)}", 
            tool_call_id=tool_call["id"]
        )
    
# Building The Summarization Chain 

def log_input(data, message):
    logger.info(f"message: {message}\n")
    return data


summarization_chain = (
    # start with initial query 
    RunnablePassthrough.assign(
        messages=lambda x: [HumanMessage(content=x["query"])]
    )
    # First LLM call (extract video ID)
    | RunnablePassthrough.assign(
        ai_response=lambda x: llm_with_tools.invoke(x["messages"])
    )
    | RunnableLambda(
        lambda x: log_input(x, f"After first LLM call. AI response: {x['ai_response']}")
    )
    | RunnablePassthrough.assign(
        tool_messages=lambda x: [
            execute_tool(tc) for tc in x['ai_response'].tool_calls
        ]
    )
    | RunnableLambda(
        lambda x: log_input(x, f"After first tool call. Tool message: {x['tool_messages'][:1000]}")
    )
    # update message history
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
    # second LLM call (fetch transcript)
    | RunnablePassthrough.assign(
        ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
    )
    | RunnableLambda(
        lambda x: log_input(x, f"Supposed to call fetch transcript tool: {x['ai_response2']}")
    )
    # Process second tool call
    | RunnablePassthrough.assign(
        tool_messages2=lambda x: [
            execute_tool(tc) for tc in x["ai_response2"].tool_calls
        ]
    )
    # Final message update
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
    )
    # Generate final summary
    | RunnablePassthrough.assign(
        summary=lambda x: llm_with_tools.invoke(x["messages"]).content
    )
    | RunnableLambda(
        lambda x: log_input(x, f"check summary: {x['summary']}")
    )
    # Return just the summary text
    | RunnableLambda(lambda x: x["summary"])
)

# Usage
result = summarization_chain.invoke({
    "query": "Summarize this YouTube video: https://www.youtube.com/watch?v=t97ipSIDEfU"
})

logging.info("Video Summary:\n", result)
# %%
