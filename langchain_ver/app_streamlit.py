import os
import tempfile
import streamlit as st
import pdfplumber
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
st.title("📄🔍 Ask Your PDF")
st.write("Upload a PDF and ask questions about its content.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# If file uploaded
if uploaded_file is not None:
    with st.spinner("Reading and processing PDF..."):

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Extract text 
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embedding_model)

        # LLM
        llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

    st.success("✅ PDF processed. You can now ask questions!")

    # Question input
    user_question = st.text_input("Ask a question about the PDF:")

    if user_question:
        with st.spinner("Thinking"):
            answer = qa_chain.run(user_question)
            st.markdown("### 📌 Answer:")
            st.write(answer)             