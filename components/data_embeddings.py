import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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

# Generate embeddings for each chunk
embedded_chunks = [{
    "title": chunk["title"],
    "source": chunk["source"],
    "content": chunk["content"],
    "embedding": embedding_model.encode(chunk["content"]).tolist()
} for chunk in chunks]

# Save the embeddings
os.makedirs("data/embeddings", exist_ok=True)
with open("data/embeddings/embedded_chunks.json", "w", encoding="utf-8") as f:
    json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks and saved to data/embeddings/embedded_chunks.json")
