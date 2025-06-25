import numpy as np
from sentence_transformers import SentenceTransformer

def retrieve_relevant_docs(query, faiss_index, model, k=3):

    """
    this function takes a query, convert query into embedding
    and find out closest k document in faiss index.
    """

    #convert user query into embedding
    query_embedding = model.encode([query])[0]

    #take closest k documents from FAISS index
    _, indices = faiss_index.search(np.array([query_embedding]), k)

    return indices