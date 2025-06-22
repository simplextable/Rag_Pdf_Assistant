import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_faiss_index(texts):
    #upload SentenceTransformers model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    #convert texts into embedding
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings = np.array(embeddings)

    #create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1]) #compare vectory by using L2 distance
    index.add(embeddings) #add embeddings into FAISS

    return index