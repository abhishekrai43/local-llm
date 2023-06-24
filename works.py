import os
import nltk
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from transformers import pipeline
import faiss


# Define the paths
gpt4all_path = './models/gpt4all-converted.bin'

# Create the callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Create the embeddings and llm objects
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

# Load the local index
index = FAISS.load_local("my_faiss_index", embeddings)

# Initialize the question-answering model
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")


# Define the prompt template
template = """
Context: {context}
Question: {question}
Answer: {answer}
"""""

# Define the similarity search function
def similarity_search(query, index, k=3):
    try:
        matched_docs = index.similarity_search(query, k=k)
        return matched_docs
    except Exception as e:
        print("An error occurred during similarity search: ", e)
        return []

# Split the documents into sentences
def split_into_sentences(document):
    return nltk.sent_tokenize(document)

# Select the best sentences based on the question
def select_best_sentences(question, sentences):
    results = []
    for sentence in sentences:
        answer = qa_model(question=question, context=sentence)
        if answer['score'] > 0.8:  # You can tune this threshold based on your requirements
            results.append(sentence)
    return results

# Answer the question
def answer_question(question):
    # Get the most similar documents
    matched_docs = similarity_search(question, index)

    # Convert the matched documents into a list of sentences
    sentences = []
    for doc in matched_docs:
        sentences.extend(split_into_sentences(doc.page_content))

    # Select the best sentences
    best_sentences = select_best_sentences(question, sentences)

    context = "\n".join([doc.page_content for doc in matched_docs])
    question = question

    # Create the prompt template
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question", "answer"])

    # Initialize the LLMChain
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    # Generate the answer
    answer = llm_chain.run(context=context,question=question,answer='', max_tokens=512, temperature=0.0, top_p=0.05)

    # Remove the template from the answer
    answer = answer.replace(template, "").strip()

    return answer

# Main loop for continuous question-answering
while True:
    # Get the user's question
    question = input("Chatbot: ")

    # Check if the user wants to exit
    if question.lower() == "exit":
        break

    # Generate the answer
    answer = answer_question(question)

    # Print the answer
    print("Answer:", answer)
