import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit Title
st.title("ChatGroq RAG with PDF")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>

Question: {input}
"""
)

# Initialize Embedding Model

# Embedding Function
def vector_embedding():
    if "vectors" not in st.session_state:
       
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_document = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )
        model_name = "sentence-transformers/all-mpnet-base-v2"
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_document, st.session_state.embeddings
        )

# UI for User Input
prompt1 = st.text_input("Enter Your Question from Documents")

# Embed Documents Button
if st.button("Document Embedding"):
    with st.spinner("Embedding documents..."):
        vector_embedding()
        st.success("Vector Store created.")

# Initialize a list in session state to store answers
if "history" not in st.session_state:
    st.session_state.history = []

# Handle Queries
if prompt1.strip():
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Please embed the documents first by clicking the 'Document Embedding' button.")
    else:
        with st.spinner("Fetching response..."):
            start = time.time()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": prompt1})
            end = time.time()

        # Store the current response and question in the history
        st.session_state.history.append({
            "question": prompt1,
            "answer": response['answer'],
            "context": response.get('context', [])
        })

        st.write(response['answer'])
        st.write(f"Response generated in {end - start:.2f} seconds.")

        # Show similar documents in the expander
        with st.expander("Document Similarity Search"):
            context = response.get('context', [])
            if not context:
                st.write("No similar documents found.")
            else:
                for i, doc in enumerate(context):
                    st.write(doc.page_content)
                    st.write("-----------------------------------------------")

# Sidebar with Chat History
with st.sidebar:
    st.header("Chat History")
    for i, entry in enumerate(st.session_state.history):
        # Display only the questions in the sidebar
        if st.button(f"Question {i + 1}: {entry['question'][:50]}..."):  # Show first 50 characters
            # Store the index of the clicked question
            st.session_state.selected_question = i

# Display the selected question's answer in the main content area
if "selected_question" in st.session_state:
    selected_entry = st.session_state.history[st.session_state.selected_question]
    st.subheader(f"Answer to: {selected_entry['question']}")
    st.write(selected_entry['answer'])
    st.write("Context:")
    for i, doc in enumerate(selected_entry["context"]):
        st.write(f"Document {i + 1}: {doc.page_content[:500]}...")  # Show a snippet of the document
