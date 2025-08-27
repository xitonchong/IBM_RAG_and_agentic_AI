


v
## Building the summarization chain

Now, you'll combine your functions into a complete `summarization_chain` using the pipe operator `|`, which applies functions sequentially (similar to function composition where `f|g(x)` is equivalent to `f(g(x))`).

The workflow follows these steps:
1. Convert the input prompt to a HumanMessage
2. Pass the message to LLM with tools
3. Extract tool calls from LLM response
4. Update message history with tool results
5. Send updated messages back to LLM
6. Repeat steps 3-5 as needed
7. Finally, extract just the content from the final message using RunnableLambda

Each step maintains state using RunnablePassthrough until you reach the final message, at which point you'll apply RunnableLambda to extract only the summary text.
