

## Download 
```bash
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/N0CceRlquaf9q85PK759WQ/regression-dataset.csv
! wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7J73m6Nsz-vmojwab91gMA/classification-dataset.csv
```


## Requirement 
```bash 
%pip install langchain-openai==0.3.10 | tail -n 1
%pip install langchain==0.3.21 | tail -n 1
%pip install openai==1.68.2 | tail -n 1
%pip install pandas==2.2.3 | tail -n 1
%pip install numpy==2.2.4 | tail -n 1
%pip install matplotlib==3.10.1 | tail -n 1
%pip install seaborn==0.13.2 | tail -n 1
%pip install scikit-learn==1.6.1 | tail -n 1

```


The tool components:

- **`@tool` Decorator:** Marks the function as a tool, allowing LangChain to integrate it and expose it to the LLM.

- **Input Arguments:** The parameters your tool function accepts, along with type annotations for clarity.

- **Tool Description:** A clear, concise explanation used by LangChain and the LLM to understand when and how to call the tool.

- **Return Type:** Specifies the type of data your tool will return, improving clarity and reliability.

- **`.name`:** Automatically derived from your Python function name; used by LangChain to identify the tool.

- **`.description`:** Automatically extracted from your function's docstring; helps the LLM understand the tool’s purpose.

- **`.args`:** Represents input arguments with their associated types, allowing LangChain to validate and pass correct values to your tool function.

Let's create the first LangChain tool, which lists all CSV files in the current directory.

- `os.getcwd()` retrieves the current working directory.
- `os.path.join(os.getcwd(), "*.csv")` constructs a path pattern to match all CSV files (`*` matches all filenames ending with `.csv`).
- `glob.glob(pattern)` returns a list of files that match the given pattern.


## Agents

Agents in LangChain are advanced components that enable AI models to decide when and how to use tools dynamically. Instead of relying on predefined scripts, agents analyze user queries and choose the best tools to achieve a goal. The next step is defining your agent, which requires specifying how it should think and behave. You'll use `ChatPromptTemplate.from_messages()` to create a structured prompt with three essential components:

1. **System message**: This establishes the agent's identity and primary objective. You define it as a data science assistant whose task is to analyze CSV files and determine whether each dataset is suitable for classification or regression based on its structure. This gives the agent a clear purpose and scope.

2. **User input**: The `{input}` placeholder will be replaced with the user's actual query. This allows the agent to respond directly to what the user is asking about.

3. **Agent scratchpad**: The `{agent_scratchpad}` placeholder is crucial for tool-calling agents as it provides space for the agent to show its reasoning process and track intermediate steps. This enables the agent to build a chain of thought, call tools sequentially, and use the results from one tool to inform decisions about subsequent tool calls.

![agents copy.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/TYkDvBmpmmSXx6TftNpJgw/agents%20copy.png)

[Reference article for image](https://medium.com/@Shamimw/understanding-langchain-tools-and-agents-a-guide-to-building-smart-ai-applications-e81d200b3c12)
