
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
import os
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from llama_cpp import Llama

 # Replace with your Hugging Face token
os.environ["HF_TOKEN"] = HF_AUTH_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_AUTH_TOKEN

# -----------------------
# Load FAISS index & metadata
# -----------------------
index = faiss.read_index("/Users/gangadhar/Documents/my_folder/my_projects/Nvidia_toolkit_SDK_bot/data/faiss/cuda_docs.index")

with open("/Users/gangadhar/Documents/my_folder/my_projects/Nvidia_toolkit_SDK_bot/data/faiss/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(query: str):
    return embedder.encode([query])[0]

def search_faiss(query, k=5):
    query_vector = embed_query(query)
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            results.append({
                "text": metadata[idx]["content"],
                "source": metadata[idx]["source"],
                "score": float(dist)
            })
    return results

def build_prompt(query, retrieved_docs):
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    prompt = f"""
You are an expert in NVIDIA SDKs.
Answer the question based only on the following context:

{context}

Question: {query}

If you don't find an answer in the context, say:
"I couldn't find relevant information in the provided context."
"""
    return prompt.strip()

# -----------------------
# Load GGUF Model (Local)
# -----------------------
print("Loading GGUF model locally...")

# Path to the downloaded GGUF file (replace with actual location)
MODEL_PATH = "/Users/gangadhar/Documents/my_folder/my_projects/Nvidia_toolkit_SDK_bot/data/saved_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=8,       # Adjust to CPU cores
    n_gpu_layers=-1,   # Use Metal GPU acceleration
    verbose=False
)

# -----------------------
# RAG Query Function
# -----------------------
def get_llm_answer(query):
    retrieved_docs = search_faiss(query, k=5)
    prompt = build_prompt(query, retrieved_docs)

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["</s>"]
    )

    return output["choices"][0]["text"]

# -----------------------
# Test Query
# -----------------------
if __name__ == "__main__":
    result = get_llm_answer("Explain the use of nvdisasm in CUDA and give an example command")
    print("\nAnswer:\n", result)
    llm.close()  # ✅ Avoids the destructor error

