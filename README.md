 NVIDIA SDK Bot — End-to-End (Scrape _ Chunk _ Embed _ FAISS _ Streamlit _ Docker _ AWS)
An end-to-end semantic search app for NVIDIA SDK documentation.
It scrapes docs, cleans & chunks text, creates embeddings with Sentence-Transformers, indexes them with FAISS, and serves results via a Streamlit UI. A Docker image makes it easy to run anywhere.
📦 What You Build
Web Scraper → pulls NVIDIA SDK pages (HTML)
Data Transformer → cleans & splits into chunks with metadata
Embeddings → encodes chunks using all-MiniLM-L6-v2
FAISS Index → builds cuda_docs.index + metadata.pkl
Streamlit App → semantic search UI
Docker Image → one-command run
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
🧰 Prerequisites
Python 3.10+
pip
(Optional) Docker
On macOS/Ubuntu, a virtual environment is recommended.
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows (PowerShell)
pip install -r requirements.txt
1) 🕸️ Scrape NVIDIA SDK Docs
Goal: Download pages you want to index.
components/web_scraper.py typically:
Takes one or more base URLs
Crawls allowed paths
Extracts main content (drops nav/footers)
Saves raw text/HTML or JSONL
Example usage (adjust flags to match your script’s arguments):
python components/web_scraper.py \
  --base-url https://docs.nvidia.com/cuda/ \
  --out data/raw/cuda_docs.jsonl \
  --max-pages 500
Output: data/raw/cuda_docs.jsonl (or similar)
2) 🧽 Clean & Chunk Text
Goal: Normalize text and split into semantic chunks with metadata (title, source URL, etc).
python components/data_transformation.py \
  --in data/raw/cuda_docs.jsonl \
  --out data/processed/chunks.jsonl \
  --chunk-size 500 \
  --chunk-overlap 50
Output: data/processed/chunks.jsonl where each row looks like:
{"title": "CUDA Toolkit Overview", "content": "…chunk text…", "source": "https://…"}
3) 🔢 Generate Embeddings
Model: sentence-transformers/all-MiniLM-L6-v2 (fast, light, great for semantic search)
python components/data_embeddings.py \
  --in data/processed/chunks.jsonl \
  --out data/processed/embeddings.npy \
  --model sentence-transformers/all-MiniLM-L6-v2
Output: embeddings.npy (shape: N × 384) matching your chunks order.
4) 📇 Build FAISS Index
Goal: Store embeddings for fast similarity search and keep metadata aligned.
python components/vector_db_embeddings.py \
  --embeddings data/processed/embeddings.npy \
  --chunks data/processed/chunks.jsonl \
  --index-out data/faiss/cuda_docs.index \
  --meta-out  data/faiss/metadata.pkl \
  --index-type ivfflat  \
  --nlist 1024
Outputs:
data/faiss/cuda_docs.index (FAISS binary)
data/faiss/metadata.pkl (Python list of dicts: {title, content, source})
5) 🖥️ Run the Streamlit App (Local)
components/app.py loads the model, FAISS index, and metadata, then runs a simple UI:
# in app.py (already in your repo)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss/cuda_docs.index")
with open("data/faiss/metadata.pkl", "rb") as f:
    chunks = pickle.load(f)
Start the app:
streamlit run components/app.py
Open http://localhost:8501.
What happens when you search:
Your query is embedded with all-MiniLM-L6-v2
FAISS index.search() returns top-K nearest chunks
The UI shows title, content snippet, source URL, and distance
🐳 Docker (Recommended for Reproducibility)
Dockerfile (key points):
Uses python:3.10-slim
Installs deps from requirements.txt
Copies the project into /app
Runs streamlit run /app/components/app.py
Build:
docker build -t cuda-docs-app .
Run (map container → host port):
docker run --rm -p 8501:8501 cuda-docs-app
# then open http://localhost:8501
If port 8501 is in use locally, use a different host port, e.g. -p 8502:8501.
✅ Quick Smoke Test (Optional)
Verify the index & metadata load correctly:
python - <<'PY'
import faiss, pickle, numpy as np
index = faiss.read_index("data/faiss/cuda_docs.index")
with open("data/faiss/metadata.pkl","rb") as f: meta = pickle.load(f)
print("Index ntotal:", index.ntotal, "Meta:", len(meta))
PY
🧩 Tips & Common Pitfalls
FAISS “vers not recognized” or read_index fails
If head data/faiss/cuda_docs.index prints lines like version https://git-lfs..., you copied a Git LFS pointer instead of the real file.
Fix: make sure the actual binary is present before building Docker (or run git lfs pull on machines that use LFS).
Missing dependencies on Linux
If you build from source, you may need build-essential, gcc, etc. The Dockerfile handles this.
Large CPU/GPU packages on small machines
Torch wheels can be large. On tiny EC2 instances disk or memory may be insufficient; Docker helps contain the environment.
Absolute paths in Docker
Inside the container, use /app/... paths (not your macOS paths). The provided Dockerfile runs streamlit run /app/components/app.py.
🔧 Configuration
Model: sentence-transformers/all-MiniLM-L6-v2 (changeable in code/flags)
Index type: IVF-Flat by default above; you can switch to Flat or HNSW depending on recall/latency needs.
Chunking: tune --chunk-size and --chunk-overlap for your docs.
🗺️ Extending the Project
Add more NVIDIA SDKs (TensorRT, cuDNN, Nsight, etc.)
Add RAG (retrieve top-K chunks, then pass to an LLM for synthesis)
Host on EC2: build + run Docker, open port 8501 in the security group
Add caching with st.cache_data for faster repeated queries
🧪 Example Code Snippets
Query from Python (without UI):
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/faiss/cuda_docs.index")
with open("data/faiss/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

q = "How do CUDA streams improve concurrency?"
qv = model.encode([q])
D, I = index.search(np.array(qv), 5)

for rank, idx in enumerate(I[0], 1):
    print(f"{rank}. {meta[idx]['title']} — {meta[idx]['source']}")