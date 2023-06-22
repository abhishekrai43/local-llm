import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Assign the path for the GPT4All model
gpt4all_path = './models/gpt4all-converted.bin'

# Callback manager for handling calls with the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create the HuggingFace embeddings object
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Create the GPT4All LLM object
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

# Load our local index vector db
index = FAISS.load_local("my_faiss_index", embeddings)

# Create the prompt template
template = """

Context: Use this context only {context}
---
Question: {question}
Answer: In the context, 
"""

# Function to handle similarity search and return the best answer
def get_best_answer(question):
    matched_docs, sources = similarity_search(question, index, n=1)
    context = "\n".join([doc.page_content for doc in matched_docs])
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(question)
    return answer

# Function to handle similarity search
def similarity_search(query, index, n=1):
    matched_docs = index.similarity_search(query, k=n)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )
    return matched_docs, sources

# Main loop for continuous question-answering
while True:
    # User input for the question
    question = input("Please enter your question (or type 'exit' to close the program): ")

    # Check if the user wants to exit the program
    if question.lower() == "exit":
        break

    # Get the best answer
    answer = get_best_answer(question)
    
    # Print the answer
    print("Answer:", answer)

# End of the program
