




def warn(*args, **kwargs): 
    pass 

import warnings 
warnings.warn = warn 
warnings.filterwarnings('ignore')

from dotenv import load_dotenv 
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter 
import wget 
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

filename = 'companyPolicies.txt'

if not os.path.exists(filename):
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
    wget.download(url, out=filename)
    print('file downloaded')



with open(filename, 'r') as file: 
    # read the context of the file
    contents = file.read() 
    # print(contents) 


loader = TextLoader(filename)
documents = loader.load() 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
texts = text_splitter.split_documents(documents) 
print(len(texts))

## Embedding and storing 
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  #store the embedding in docsearch using chromadb
print('document ingested')


prompt_template = """
    Use the information from the document to answer the question,
    if you do not know, answer you don't know. definitely dont try 
    to make up answer. 

    {context} 

    Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

## Indexing 
''' langchain is used to split the document and create chunks. 
    For the splitting process, the goal is to ensure that each segment is as extensive 
    as if you were to count to a certain number of characters nand meet the split separator.
    let's set 1000 as the chunk size in this project. though the chunk size is 1000, the spitting 
    is happening randomly. this is an issue with langchain. characterTextSplitter 
    uses \n\n as the deafult spllit separator. you can change it by addding the separator 
    parameter in CaracterTextSplitter function; for example, separator='\n'
'''

## LLM model construction
# model_id = 'gemini-1.5-flash'
model_id = 'gemini-1.5-flash'
llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.5)

qa = RetrievalQA.from_chain_type(llm=llm
    , chain_type='stuff'
    , chain_type_kwargs=chain_type_kwargs
    , retriever=docsearch.as_retriever())

query = "Can I eat in company vehicles? "
response = qa.invoke(query) 
print(response) 


## Make the conversation have memory 
'''
Do you want your conversations with an LLM to be more like a dialogue with 
a friend? 
'''

memory = ConversationBufferMemory(memory_key="chat_history", return_mesage=True)

qa = ConversationalRetrievalChain.from_llm(llm=llm
        , chain_type="stuff"
        , retriever=docsearch.as_retriever() 
        , memory = memory 
        , get_chat_history = lambda h: h
)

# create a history list to store the chat history 
history = []
query = "what is mobile policy?"
result = qa.invoke({"question": query, "chat_history": history})
print(result["answer"])

# append the previous query and answer to the history 
history.append((query, result['answer']))

query = "Summarize into list of points"
result = qa({"question": query}, {"chat_history": history})

print(result["answer"])

# append the previous and answer to the chat history again 
history.append((query, result["answer"])) 

query = "what is the aim of it?"
result = qa({"question": query}, {"chat_history": history})

print(result["answer"])


def qa():
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                               chain_type="stuff", 
                                               retriever=docsearch.as_retriever(), 
                                               memory = memory, 
                                               get_chat_history=lambda h : h, 
                                               return_source_documents=False)
    history = []
    while True:
        query = input("Question: ")
        
        if query.lower() in ["quit","exit","bye"]:
            print("Answer: Goodbye!")
            break
            
        result = qa({"question": query}, {"chat_history": history})
        
        history.append((query, result["answer"]))
        
        print("Answer: ", result["answer"])


qa()