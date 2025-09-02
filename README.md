 NVIDIA SDK Bot — End-to-End (Scrape _ Chunk _ Embed _ FAISS _ Streamlit _ Docker _ AWS)

 
An end-to-end semantic search app for NVIDIA SDK documentation.
It scrapes docs, cleans & chunks text, creates embeddings with Sentence-Transformers, indexes them with FAISS, and serves results via a Streamlit UI. A Docker image makes it easy to run anywhere.


. Web Scraper → pulls NVIDIA SDK pages (HTML)
. Data Transformer → cleans & splits into chunks with metadata
. Embeddings → encodes chunks using all-MiniLM-L6-v2
. FAISS Index → builds cuda_docs.index + metadata.pkl
. Streamlit App → semantic search UI
. Docker Image → one-command run
. AWS - Deployment in EC2 instance

🗂️ Repository Layout
AI-Assisted-Learning-for-NVIDIA-SDKs-and-Toolkits/
├─ components/
│  ├─ app.py                         # Streamlit UI
│  ├─ web_scraper.py                 # Scrape NVIDIA docs
│  ├─ data_transformation.py         # Clean & chunk text
│  ├─ data_embeddings.py             # Create sentence embeddings
│  ├─ vector_db_embeddings.py        # Build FAISS index
│  └─ RAG_pipeline.py                # (Optional) retrieval helpers
├─ data/
│  └─ faiss/
│     ├─ cuda_docs.index             # FAISS index (binary)
│     └─ metadata.pkl                # [{title, content, source}, ...]
├─ requirements.txt
├─ Dockerfile
└─ README.md
Note: data/faiss/cuda_docs.index and data/faiss/metadata.pkl are generated artifacts. You can pre-build them locally and commit, or rebuild them from scratch using the steps below.


1) Scrape NVIDIA SDK Docs
Goal: Download wanted pages from web.
2) Clean & Chunk Text
Goal: Normalize text and split into semantic chunks with metadata (title, source URL, etc).
3) Generate Embeddings
Model: sentence-transformers/all-MiniLM-L6-v2 (fast, light, great for semantic search)
Output: embeddings.npy
4) Build FAISS Index
Goal: Store embeddings for fast similarity search and keep metadata aligned.
5) Run the Streamlit App (Local)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss/cuda_docs.index")
with open("data/faiss/metadata.pkl", "rb") as f:
    chunks = pickle.load(f)
6) Docker 


