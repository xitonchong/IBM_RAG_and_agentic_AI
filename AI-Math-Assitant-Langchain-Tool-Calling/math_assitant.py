from langchain.agents import AgentType
from langchain_community.llms.ollama import Ollama
import re
import ollama
from icecream import ic 
from langchain_core.tools import tool 
from typing import List, Dict, Union 
from langchain.agents import initialize_agent 
from custom_tools import (
    subtract_numbers
    , add_numbers
    , add_numbers_with_options
    , sum_numbers_with_complex_output
    , multiply_numbers
    , divide_numbers
    , new_subtract_numbers
)
llm = Ollama(model="llama3.2")

# response = llm.invoke("What is tool calling in langchain?")
# ic(response)
# print("\nResponse Content: ", response)


## Function 
'''
In AI, a tool will call a basic function or capability that can be called on to perform 
a specific task. think of it like a single item in a toolbox;

When building tools for tool calling, there are a few key principle to keep in mind 
1. Clear purpose. make sure the tool has a well-defined job. 
2. Standardized input. The tool should accept input in a predictable, structured format so it's easy to use
3. Consistent output. Always return results in a format that's easy to process or integrate with other systems.
4. Comprehensive documentation.  Your tool should include clear, simple documentation that explains what it does, how to use it,
    and any quircks or limitations.

'''


output = add_numbers("1 2")
ic(output)

## Tool 
'''
The tool class in LangChain serves as a structured wrapper that converts regular python 
functions into agent-compatible tools. each tool needs 3 key components. 
1. name 
2. function that performs the actual operation 
3. description

'''

from langchain.agents import Tool 

add_tool = Tool(
    name = "AddTool", 
    func = add_numbers, 
    description="Adds a list of numbers and returns the result."
)


print("tool object", add_tool) 

print("Tool Name: ")
print(add_tool.name) 


print("Tool description")
print(add_tool.description)


print("Tool function")
print(add_tool.invoke)





print("Name: \n", add_numbers.name)
print("Description: \n", add_numbers.description) 
print("Args: \n", add_numbers.args) 

test_input = "what is the sum between 10, 20 and 30 " 
print(add_numbers.invoke(test_input))  # Example


## Ue @tool-StructuredTool 
'''
The @tool decorator creates a StructuredTool with schema information extracted from 
function signatures and docstrings as shown here. this helps LLMs better understand what 
inputs the tool expects and how to use it properly. While both approaches work, 
@tool is generally preferred for modern LangChain app, esp with LangGraph andn function-calling models.

'''
# Comparing the two approaches
print("Tool Constructor Approach:")

print(f"Has Schema: {hasattr(add_tool, 'args_schema')}")
print("\n")

print("@tool Decorator Approach:")


print(f"Has Schema: {hasattr(add_numbers, 'args_schema')}")
print(f"Args Schema Info: {add_numbers.args}")




### let's compare the arguments for addd_numbers_with_options and add_numbers;
### Tool not able to see the optional behavior while @tool can handle that


print(f"Args Schema Info: {add_numbers_with_options.args}")
print(f"Args Schema Info: {add_numbers.args}")



print(add_numbers_with_options.invoke({"numbers": [-1.1, -2.1, -3.0], "absolute": False}))
print(add_numbers_with_options.invoke({"numbers":[-1.1,-2.1,-3.0],"absolute":True}))



    

@tool
def sum_numbers_from_text(inputs: str) -> float:
    """
    Adds a list of numbers provided in the input string.
    
    Args:
        text: A string containing numbers that should be extracted and summed.
        
    Returns:
        The sum of all numbers found in the input.
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return result

##::Initialize_agent 
"""
relationship bewteen agent and LLM 
- agent acts as the decision maker, figuring out which tools and an LLM to work together 
- LLM is the reasoning engine. 
"""

agent = initialize_agent([add_tool], llm, agent='zero-shot-react-description',
                 verbose=True, handle_parsing_errors=True)


response = agent.run("In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total.")


print(response)

response = agent.invoke({"input": "Add 10, 20, two and 30"})
print(response) 


## Structured chat zero shot react-description 
'''
Agents like zero-shot-react-description expect tools to take and return plan strings, 
which works weel with manully defined Tool(...) wrappers.

in constrast, structured agents like structured-chat-zero-shot-react-description or 
openai-functions are built to handle structured inputs and outputs via @tool decorator. 
if a tool returns a dictionary but the agent expects a string, it can cause key error 
or parsing failures. 

in the agent example below, use sum_numbers_from_text as a tool and structured-chat
zero-shot-react-description as the agent type
'''
agent_2 = initialize_agent([sum_numbers_from_text], llm, 
        agent="structured-chat-zero-shot-react-description", verbose=True, handle_parsing_errors=True)
response = agent_2.invoke({"input": "Add 10, 20 and 30"})
print(response)


llm_ai = Ollama(model="gpt-oss:20b")

agent_3 = initialize_agent([sum_numbers_with_complex_output],llm_ai,
                agent="openai-functions", verbose=True, 
                handle_parsing_errors=True
)

response = agent_3.invoke({"input": "Add 10, 20 and 30"})
print(response)



# Tools with multiple inputs. 
# some models cannot handle complex output parsing reliably, especially when used
# with agents like 'structured-chat-zero-shot-react-description'. 



print('intialize agent as structured-chat-zero-shot-react-description`')

agent2 = initialize_agent(
    [add_numbers_with_options], 
    llm_ai,  # chat-gpt-oss:20b
    agent="structured-chat-zero-shot-react-description",
    verbose=True
)

response = agent2.invoke({
    "input": "Add -10, -20 and -30 using absolute values."
})
print(response) 

# let;s try with gps-oss:20b and see if runs with multiple inputs 
# actually it didn;t manage to output any values,
#  {'input': 'Add -10, -20 and -30 using absolute values.', 'output': ''}


## Create reac agent
from langgraph.prebuilt import create_react_agent 
from langchain_openai import ChatOpenAI 
from openai import OpenAI 
from langchain_ollama import ChatOllama
 

# client = OpenAI(
#     base_url = 'http://localhost:11434/v1', 
#     api_key="ollama" # dumy key
# )

# openai_llm = ChatOpenAI(model='gpt-4o-mini')
# openai_llm_with_tools = openai_llm.bind([sum_numbers_from_text])


# Ollama does not implemetn bind_tools 
llm_ai = ChatOllama(model="gpt-oss:20b")
llm_ai_with_tools = llm_ai.bind_tools([sum_numbers_from_text])


agent_exec = create_react_agent(model=llm_ai_with_tools,
                tools=[sum_numbers_from_text]                                
)
msgs = agent_exec.invoke({"messages": [("human", "add the numbers -10, -20, -30")]})
print(msgs["messages"][-1].content)



## Subtraction tools 
print("Name: \n", subtract_numbers.name) 
print("description: \n", subtract_numbers.description)
print("args: \n", subtract_numbers.args)

print("Calling Tool Function:")
test_input = "10 20 30 and four a b" 
print(subtract_numbers.invoke(test_input))  # Example



# Testing multiply_tool
multiply_test_input = "2, 3, and four "
multiply_result = multiply_numbers.invoke(multiply_test_input)
print("--- Testing MultiplyTool ---")
print(f"Input: {multiply_test_input}")
print(f"Output: {multiply_result}")


# Testing divide_tool
divide_test_input = "100, 5, two"
divide_result = divide_numbers.invoke(divide_test_input)
print("--- Testing DivideTool ---")
print(f"Input: {divide_test_input}")
print(f"Output: {divide_result}")


# the agent is using the tools correctly, non-numeric numbers repr is not translated


# Build the agent - Comining all the tools together 
tools = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers]
llm_ai_with_tools = llm_ai.bind_tools(tools)


from langgraph.prebuilt import create_react_agent 


# create the agent wiht tools 
math_agent = create_react_agent( 
    model=llm_ai_with_tools, 
    tools=tools,
    ## Optional: add a system message to guide the agent's behavior 
    prompt="You are a helpful mathematcial assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly."
)

response = math_agent.invoke({
    "messages": [("human", "What is 25 divided by 4?")]
})
final_answer = response["messages"][-1].content 
ic(final_answer) 


print('\n', '='*100)
response_2 = math_agent.invoke({
    "messages": [("human", "Subtract 100, 20, and 10.")]
})

# Get the final answer
final_answer_2 = response_2["messages"][-2].content
print(final_answer_2)


tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers]
llm_ai_with_tools = llm_ai.bind_tools(tools_updated)

# Create the agent with all tools
math_agent_new = create_react_agent(
    model=llm_ai_with_tools,
    tools=tools_updated,
    # Optional: Add a system message to guide the agent's behavior
    prompt="You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly."
)
print("agent",math_agent_new)



# Test Cases
test_cases = [
    {
        "query": "Subtract 100, 20, and 10.",
        "expected": {"result": 70},
        "description": "Testing subtraction tool with sequential subtraction."
    },
    {
        "query": "Multiply 2, 3, and 4.",
        "expected": {"result": 24},
        "description": "Testing multiplication tool for a list of numbers."
    },
    {
        "query": "Divide 100 by 5 and then by 2.",
        "expected": {"result": 10.0},
        "description": "Testing division tool with sequential division."
    },
    {
        "query": "Subtract 50 from 20.",
        "expected": {"result": -30},
        "description": "Testing subtraction tool with negative results."
    }

]


correct_tasks = []
# Corrected test execution
for index, test in enumerate(test_cases, start=1):
    query = test["query"]
    expected_result = test["expected"]["result"]  # Extract just the value
    
    print(f"\n--- Test Case {index}: {test['description']} ---")
    print(f"Query: {query}")
    
    # Properly format the input
    response = math_agent_new.invoke({"messages": [("human", query)]})
    
    # Find the tool message in the response
    tool_message = None
    for msg in response["messages"]:
        if hasattr(msg, 'name') and msg.name in ['add_numbers', 'new_subtract_numbers', 'multiply_numbers', 'divide_numbers']:
            tool_message = msg
            break
    
    if tool_message:
        # Parse the tool result from its content
        import json
        tool_result = json.loads(tool_message.content)["result"]
        print(f"Tool Result: {tool_result}")
        print(f"Expected Result: {expected_result}")
        
        if tool_result == expected_result:
            print(f"✅ Test Passed: {test['description']}")
            correct_tasks.append(test["description"])
        else:
            print(f"❌ Test Failed: {test['description']}")
    else:
        print("❌ No tool was called by the agent")

print("\nCorrectly passed tests:", correct_tasks)