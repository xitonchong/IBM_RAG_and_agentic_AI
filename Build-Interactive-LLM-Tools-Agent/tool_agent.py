from langchain_core.tools import Tool 
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama 
from langchain_core.tools import tool 
from my_tools import * 
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


llm = ChatOllama(model="gpt-oss:20b")

tool_map = {
    "add": add, 
    "subtract": subtract, 
    "multiply": multiply
}


tools = [add]


class ToolCallingAgent: 
    def __init__(self, llm: ChatOllama): 
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = tool_map

    def run(self, query: str) -> str: 
        # step 1: initial user message 
        chat_history = [HumanMessage(content=query)]

        # step 2: LLLM chosses tool 
        response = self.llm_with_tools.invoke(chat_history) 
        logging.info(response) 
        if not response.tool_calls: 
            return response.content 
        # step 3: handle first tool call 
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        # step 4: call tool manually 
        tool_result = self.tool_map[tool_name].invoke(tool_args) 
        logging.info(f"tool result: {tool_result}")

        # step 5: send result back to LLM 
        tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_call_id) 
        chat_history.extend([response, tool_message])

        # step 6: final LLM result 
        final_response = self.llm_with_tools.invoke(chat_history) 
        return final_response.content 
    

tool_agent= ToolCallingAgent(llm) 


print(tool_agent.run("one plus 2 "))
print(tool_agent.run("one - 2"))
print(tool_agent.run("three times two"))



## Conclusion
"""
- structure user interaction and setup chat models for real-time, context-aware conversations
- extract tool names and arguments to precisly match user intent 
- parse complex tool instructions, including handling multiple toll calls 
- build and refine an agent class to automate the entire tool-calling process 
- demonntate how these componentns work together to transform LLMs from passive responders to intelligent agents. 
"""

tool_map['calculate_tip'] = calculate_tip
##Section: Exercise 
query = "How much should I tip on $60 at 20%"

class TipAgent: 
    def __init__(self, llm: ChatOllama): 
        self.llm_with_tools= llm.bind_tools([calculate_tip])

    def run(self, query: str) -> str: 
        chat_history = [HumanMessage(content=query)]
        response = self.llm_with_tools.invoke(chat_history) 
        if not response.tool_calls: 
            return response.content 
        tool_calls = response.tool_calls
        tool_name = tool_calls[0]["name"]
        tool_args = tool_calls[0]["args"]
        tool_call_id = tool_calls[0]["id"]

        tool_response = tool_map[tool_name].invoke(tool_args) 
        tool_message = ToolMessage(content=tool_response, 
                tool_call_id = tool_call_id
        )
        chat_history.extend([response, tool_message])
        return self.llm_with_tools.invoke(chat_history).content
    

agent = TipAgent(llm) 
agent.run(query) 