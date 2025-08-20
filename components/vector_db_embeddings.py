import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Load environment variables
load_dotenv()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "!", ".", ",", " ", ""]
)

chunks = []

# Read the cleaned CUDA docs and split into chunks
with open("data/transformed_web_scraped_docs/cleaned_cuda_docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)
    for doc in docs:
        for chunk in text_splitter.split_text(doc["content"]):
            chunks.append({
                "title": doc["title"],
                "content": chunk,
                "source": doc["source"]
            })

# Load local embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings (store them in a NumPy array)
embeddings = np.array([embedding_model.encode(chunk["content"]) for chunk in chunks])

# Create FAISS index (L2 similarity)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

# Add embeddings to index
index.add(embeddings)

# Save FAISS index
os.makedirs("data/faiss", exist_ok=True)
faiss.write_index(index, "data/faiss/cuda_docs.index")

# Save metadata (chunks info)
with open("data/faiss/metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Stored {len(chunks)} chunks in FAISS index at data/faiss/cuda_docs.index")
