import streamlit as st
from sentence_transformers import SentenceTransformer
from pdf_processing import extract_text_from_pdf
from faiss_index import create_faiss_index
from retrieval import retrieve_relevant_docs
from gpt_integration import get_gpt_answer

#Upload embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

uploaded_file = st.file_uploader("Upload PDF File", type="pdf")
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    #Parse text from PDF
    pdf_text = extract_text_from_pdf("uploaded.pdf") 

    #Create FAISS Index
    fais_index = create_faiss_index([pdf_text])

    #User Question
    user_question = st.text_input("You can ask a question:")

    if user_question:
        #By using query, get closest documents
        indices = retrieve_relevant_docs(user_question, fais_index, model)
        relevant_text = pdf_text #pdf_text[indices[0]] #get closest document

        #Get answer from gpt 3.5
        answer = get_gpt_answer(relevant_text, user_question)
        st.write(answer)