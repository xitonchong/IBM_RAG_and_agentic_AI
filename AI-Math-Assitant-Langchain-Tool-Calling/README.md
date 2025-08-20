### **`create_react_agent`**

As LangChain's `AgentExecutor` is being deprecated, create_react_agent from LangGraph provides a more flexible and powerful alternative for building AI agents. This function creates a graph-based agent that works with chat models and supports tool-calling functionality.

---

#### **Key parameters of `create_react_agent`**

1. **`model`**
    - The language model that powers the agent's reasoning.
    - Must support tool calling for full functionality.

2.  **`tools`**
    - A list of tools the agent can use to perform actions.
    - Can be LangChain tools, Python functions with @tool decorator, or a ToolNode instance
    - Each tool should have a name, description, and implementation

3. **`prompt (optional)`**:
   - Customizes the instructions given to the LLM
   - Can be:
        - A string (converted to a SystemMessage)
        - A SystemMessage object
        - A function that transforms the state
        - A Runnable that processes the state

and other parameters. To see more parameters, see [docs](https://langchain-ai.github.io/langgraph/reference/prebuilt/).

#### How it works

Unlike the legacy `AgentExecutor`, which used a fixed loop structure, `create_react_agent` creates a graph with these key nodes:

1. **Agent Node:** Calls the LLM with the message history
2. **Tools Node:** Executes any tool calls from the LLM's response
3. **Continue/End Nodes:** Manage the workflow based on whether tool calls are present

The graph follows this process:

1. User message enters the graph
2. LLM generates a response, potentially with tool calls
3. If tool calls exist, they're executed and their results are added to the message history
4. The updated messages are sent back to the LLM
5. This loop continues until the LLM responds without tool calls
6. The final state with all messages is returned