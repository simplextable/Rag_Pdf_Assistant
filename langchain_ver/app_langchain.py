import os
import pdfplumber
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

#Load environment variables

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

open_api_key = os.getenv("OPENAI_API_KEY")

#1. Extract the Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

#2. Read the Text in PDF
pdf_text = extract_text_from_pdf("example.pdf")

#3. Chunk
splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(pdf_text)

#4. SentenceTransformer embedding
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#5. Create FAISS Index
vectorstore = FAISS.from_texts(chunks,embedding_model)

#6. Run ChatGPT Model
llm = ChatOpenAI(api_key=open_api_key, model_name="gpt-3.5-turbo")

#7. Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

#8. Ask Question 
question = "What kind of experience does this person have?"
answer = qa_chain.run(question)

print("Answer:\n", answer)