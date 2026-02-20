# Enterprise Insight Engine

Enterprise Insight Engine is a document-driven analytical reasoning system that enables users to:

- Upload enterprise documents (PDF, DOCX, CSV, XLSX, TXT, JSON)
- Store them into a vector database (ChromaDB)
- Ask questions via a chat interface
- Retrieve answers generated using a local LLM, grounded only in retrieved document context

This project uses:
- **Streamlit** (Frontend UI)
- **FastAPI** (Backend API)
- **ChromaDB** (Vector Store)
- **HuggingFace Embeddings** (`intfloat/e5-base-v2`)
- **Local LLM Inference** (`microsoft/Phi-3-mini-4k-instruct`)

---

## 1) Overview

The system follows a Retrieval-Augmented Generation (RAG) workflow:

1. User uploads documents
2. Documents are chunked and embedded
3. Embeddings are stored in ChromaDB
4. User asks a question
5. Question is embedded and matched against stored chunks
6. Top chunks are passed into the LLM as context
7. LLM generates an answer using only that context

This ensures answers are based on uploaded enterprise documents instead of random model knowledge.

---

## 2) Architecture

### High-Level Architecture Diagram (Text)

```

┌───────────────────────────┐
│        Streamlit UI        │
│  (app.py + chat_UI.py +    │
│   upload_docs_UI.py)       │
└──────────────┬────────────┘
│ HTTP (REST)
▼
┌───────────────────────────┐
│        FastAPI API         │
│        (backend.py)        │
│  /upload   /query          │
└──────────────┬────────────┘
│
▼
┌───────────────────────────┐
│   Document Processing      │
│   (doc_embedding.py)       │
│ - load docs                │
│ - chunk text               │
│ - embed chunks             │
└──────────────┬────────────┘
│
▼
┌───────────────────────────┐
│     Chroma Vector DB       │
│  (persisted on disk)       │
└──────────────┬────────────┘
│
▼
┌───────────────────────────┐
│     Query + Generation     │
│    (query_vectorDB.py)     │
│ - similarity search        │
│ - build prompt             │
│ - local LLM inference      │
└───────────────────────────┘

````

---

## 3) Components Breakdown

### Streamlit Frontend
- `app.py`  
  Controls page layout, tabs, styling, and app title.

- `upload_docs_UI.py`  
  Handles uploading documents to backend via `/upload`.

- `chat_UI.py`  
  Provides chat interface, stores conversation history in session state, sends queries to `/query`.

---

### FastAPI Backend
- `backend.py`
  Provides 2 endpoints:
  - `POST /upload` → processes and stores documents in vector DB
  - `POST /query` → performs retrieval + LLM generation

---

### Vector + RAG Logic
- `doc_embedding.py`
  Loads documents, chunks them, generates embeddings, stores them in ChromaDB.

- `query_vectorDB.py`
  Embeds user query, retrieves top chunks from ChromaDB, builds a prompt, generates response with local LLM.

---

## 4) Model Choices

### Embedding Model: `intfloat/e5-base-v2`
Chosen because:
- Strong semantic search performance
- Works well for enterprise documents
- Efficient enough to run locally
- Produces high-quality dense embeddings

---

### LLM Model: `microsoft/Phi-3-mini-4k-instruct`
Chosen because:
- Small enough to run locally on CPU
- Good instruction-following ability
- Fast compared to larger LLMs
- Supports enterprise-style summarization and reasoning

---

### Vector Database: ChromaDB
Chosen because:
- Simple local persistence
- No need for external cloud services
- Fast similarity search
- Works well with LangChain

---

## 5) Limitations

This system currently has the following limitations:

### 1. Chunking Configuration
In `doc_embedding.py`, chunk size is currently:

```python
chunk_size=10
chunk_overlap=5
````

This is too small and reduces retrieval quality.
It creates too many tiny chunks and breaks meaning.

---

### 2. LLM Output Truncation

In `query_vectorDB.py`:

```python
max_tokens: int = 20
```

This will cut answers too early and make outputs incomplete.

---

### 3. Prompt Leakage

The current pipeline may return parts of the prompt along with the answer because:

```python
outputs[0]["generated_text"]
```

includes the full generated sequence (prompt + output).

---

### 4. No Document Deletion

Once a document is embedded into ChromaDB, there is no feature to delete it.

---

### 5. Duplicate Detection Based Only on File Name

Duplicate checking is done only by file name.
If the same file is uploaded with a different name, it will be stored again.

---

### 6. CPU-Only Inference

Your pipeline uses:

```python
device=-1
```

So the LLM runs on CPU. It will be slow for long answers.

---

### 7. Multi-User Persistence Issues

Streamlit session state stores conversations per browser session.
If multiple users use it, chat history is not stored permanently.

---

## 6) Future Improvements

Recommended improvements for production-level usage:

### Retrieval Improvements

* Use better chunking rules per file type
* Add metadata filtering (e.g., by file, date, category)
* Add re-ranking (cross encoder) for better accuracy

### LLM Improvements

* Enable GPU inference
* Increase token output safely
* Return only answer text (strip prompt)

### Application Improvements

* Add document delete + re-index features
* Add authentication
* Add chat history persistence (SQLite/PostgreSQL)
* Add response streaming in chat UI
* Add progress indicators for embedding stage

### Data Safety Improvements

* Add validation and file scanning
* Add maximum upload size
* Add rate limiting for API calls

---

## 7) Setup Instructions

### Step 1: Clone Repository

```bash
git clone <your_repo_url>
cd <your_project_folder>
```

---

### Step 2: Create Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 8) Environment Variables (.env)

Create a `.env` file in the project root:

```env
CHROMA_DB_PATH=Chroma
similarity_search_index=5
SIMILARITY_MARGIN_VALUE=0.4
```

### Meaning

* `CHROMA_DB_PATH`: Folder where Chroma stores vectors
* `similarity_search_index`: Number of top retrieved results
* `SIMILARITY_MARGIN_VALUE`: Minimum relevance score allowed

---

## 9) How to Run the App

You need to run **backend** and **frontend** in two terminals.

---

### Terminal 1: Start FastAPI Backend

```bash
uvicorn backend:app --reload --port 8000
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### Terminal 2: Start Streamlit Frontend

```bash
streamlit run app.py
```

Frontend runs at:

```
http://localhost:8501
```

---

## 10) Example API Calls

### Upload Documents Endpoint

**Endpoint**

```
POST /upload
```

**Example using curl**

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@sample.pdf" \
  -F "files=@budget.xlsx"
```

**Example response**

```json
{
  "status": "success",
  "duplicates": []
}
```

If duplicates exist:

```json
{
  "status": "duplicate",
  "duplicates": ["sample.pdf"]
}
```

---

### Query Endpoint

**Endpoint**

```
POST /query
```

**Example using curl**

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the total project cost?\"}"
```

**Example response**

```json
{
  "status": "success",
  "response": "The total project cost is 2,450,000 LKR.\n\n📂 Sources Utilized:\n    • budget.xlsx"
}
```

---

## 11) Assumptions

This system is built based on these assumptions:

1. Users upload enterprise documents containing useful text content.
2. Uploaded documents are safe and trusted (no malware scanning currently).
3. The vector DB is local and persists in the machine running the backend.
4. Document processing is done per upload batch.
5. The system is intended for small-to-medium datasets (not millions of chunks).
6. Query responses must be grounded in retrieved context only.

---

## 12) Trade-offs

### Local Inference vs Cloud LLM

✅ Pros:

* Works offline
* No API cost
* More privacy for enterprise documents

❌ Cons:

* Slower on CPU
* Lower quality compared to large cloud models
* Limited context window

---

### ChromaDB Local Storage vs Cloud Vector DB

✅ Pros:

* Easy setup
* No external dependency
* Fast local search

❌ Cons:

* Not ideal for multi-user cloud deployment
* Scaling limitations

---

### Duplicate Detection by File Name

✅ Pros:

* Very fast and simple

❌ Cons:

* Does not detect duplicate content with different file names

---

## 13) Known Issues

* Very small chunk size reduces answer quality
* Very low token output truncates answers
* Prompt may appear inside response
* No UI feature to clear/reset vector DB

---

## 14) License

This project is currently for educational and internal use.

---

## 15) Credits

* Streamlit
* FastAPI
* HuggingFace Transformers
* LangChain
* ChromaDB

```

---

If you want, I can also generate for you:
✅ a proper `.gitignore`  
✅ a `run_backend.bat` + `run_frontend.bat` for one-click running on Windows  
✅ improved code fixes (chunk size, token output, prompt stripping)
```
