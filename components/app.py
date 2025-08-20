import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model, FAISS index, and metadata
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss/cuda_docs.index")

with open("data/faiss/metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

# Streamlit UI
st.title("🔍 CUDA Docs Search (FAISS + MiniLM)")

query = st.text_input("Enter your search query:")
top_k = st.selectbox("Select top K results", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=4)

if query:
    # Embed query
    query_vector = embedding_model.encode([query])
    
    # Search FAISS index
    distances, indices = index.search(np.array(query_vector), top_k)
    
    st.subheader("Results:")
    for i, idx in enumerate(indices[0]):
        st.markdown(f"**{i+1}. {chunks[idx]['title']}**")
        st.write(chunks[idx]["content"])
        st.caption(f"📖 Source: {chunks[idx]['source']} | Distance: {distances[0][i]:.4f}")
