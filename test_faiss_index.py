from faiss_index import create_faiss_index
from sentence_transformers import SentenceTransformer
import numpy as np

def test_faiss_index():
    #example sentences for test
    texts = [
        "Python is a programming language.",
        "FAISS is a library for similarity search.",
        "Sentence transformers convert sentences to embeddings.",
        "Embedding vectors help with information retrieval.",
        "Deep learning models are powerful tools for AI."
    ]

    #FAISS index creation
    index = create_faiss_index(texts)

    #import SentenceTransformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    #query data
    query = "What is FAISS used for"

    #lets query inside FAISS index
    query_embedding = model.encode([query])[0]
    _, indices = index.search(np.array([query_embedding]), k=3) #bring closest 3 text

    #check whether returned indexes are working right or not
    print("Top 3 nearest documents for the query:")
    for idx in indices[0]:
        print(texts[idx]) 




if __name__ == "__main__":
    test_faiss_index()