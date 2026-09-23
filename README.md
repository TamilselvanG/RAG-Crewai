# RAG Assistant — CrewAI + FastAPI + Streamlit

A full-stack Retrieval-Augmented Generation (RAG) application. Upload PDF / TXT / MD documents through a Streamlit UI, index them into a local ChromaDB vector store, and ask questions answered by a CrewAI multi-agent pipeline running on GPT-4o-mini.

---

## Project Structure

```
RAG-Crewai/
├── README.md
├── requirements.txt
├── backend/
│   ├── main.py          # FastAPI app: /upload, /query, /health
│   └── rag_engine.py    # Ingestion, vector store, CrewAI agents & tools
└── frontend/
    └── app.py           # Streamlit chat UI
```

At runtime the backend also creates:

- `backend/uploads/` — temporary landing spot for uploaded files (deleted after ingestion).
- `backend/chroma_db/` — persistent ChromaDB vector store.

---

## Architecture

```
┌───────────────┐   HTTP    ┌──────────────────┐    ┌────────────────────┐
│  Streamlit UI │ ────────► │  FastAPI backend │ ──►│  rag_engine.py     │
│  frontend/    │  /upload  │  backend/main.py │    │  - Chunk & embed   │
│  app.py       │  /query   │                  │    │  - ChromaDB store  │
└───────────────┘ ◄──────── └──────────────────┘    │  - CrewAI agents   │
                   answer                            └────────────────────┘
```

### Components

- **Frontend (`frontend/app.py`)** — Streamlit chat app. Sidebar handles file upload and an "Enable Web Search" toggle; the main pane is a chat that POSTs to the FastAPI backend.
- **Backend API (`backend/main.py`)** — FastAPI service exposing:
  - `POST /upload` — accepts `.pdf`, `.txt`, `.md`; saves temporarily, chunks and embeds into ChromaDB, then removes the temp file.
  - `POST /query` — body `{ "query": str, "enable_web_search": bool }`; runs the CrewAI pipeline and returns `{ "answer": str }`.
  - `GET /health` — liveness probe.
- **RAG engine (`backend/rag_engine.py`)**:
  - **Loaders**: `PyPDFLoader` for PDFs, `TextLoader` for TXT/MD.
  - **Splitter**: `RecursiveCharacterTextSplitter` (chunk size 800, overlap 100).
  - **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (CPU, normalized).
  - **Vector store**: local persistent `Chroma` collection `rag_documents` at `backend/chroma_db/`.
  - **LLM**: CrewAI `LLM` wrapping `gpt-4o-mini` (temperature `0.2`).
  - **Tool**: `Search Vector Database` — top-k=4 similarity search over the Chroma store.
  - **Agents (sequential Crew)**:
    1. **Information Retrieval Specialist** — uses the vector-search tool to gather factual excerpts.
    2. **Knowledge Synthesizer & Writer** — composes a grounded markdown answer from those excerpts.

---

## Prerequisites

- Python 3.10+
- An OpenAI API key (used by CrewAI for the `gpt-4o-mini` LLM)

---

## Setup

1. **Clone and enter the project**

   ```bash
   git clone <repository-url>
   cd RAG-Crewai
   ```

2. **Create and activate a virtual environment**

   ```powershell
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   ```bash
   # Linux / macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```dotenv
   OPENAI_API_KEY=sk-...
   # Optional — override backend URL used by the Streamlit frontend
   FASTAPI_BACKEND_URL=http://localhost:8080
   ```

---

## Running the App

Open two terminals (both with the virtualenv activated).

**Terminal 1 — start the FastAPI backend:**

```bash
cd backend
uvicorn main:app --reload --port 8080
```

The API listens on `http://localhost:8080` (auto-reload enabled). Swagger UI is available at `http://localhost:8080/docs`.

**Terminal 2 — start the Streamlit frontend:**

```bash
cd frontend
streamlit run app.py
```

Streamlit will open in your browser (default `http://localhost:8501`).

### Using the UI

1. In the sidebar, upload a PDF / TXT / MD file and click **Index Document**.
2. Toggle **Enable Web Search** as desired (see note below).
3. Ask questions in the chat input. Responses come back from the CrewAI agents.
4. **Clear Chat History** resets the conversation state.

---

## Running with Docker

The project ships with `Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml` at the project root.

### Prerequisites

- Docker Engine 24+ and Docker Compose v2
- A `.env` file in the project root with your OpenAI key:

  ```dotenv
  OPENAI_API_KEY=sk-...
  ```

  `FASTAPI_BACKEND_URL` is set automatically by Compose for the frontend container — you do not need to define it here.

### Build & start both services

```bash
docker compose up --build
```

- Frontend (Streamlit): `http://localhost:8501`
- Backend (FastAPI): `http://localhost:8080` — Swagger UI at `http://localhost:8080/docs`

Run detached:

```bash
docker compose up --build -d
docker compose logs -f
```

### Stop

```bash
docker compose down          # keep the Chroma vector store
docker compose down -v       # also delete the chroma_data volume (resets the index)
```

### Build the images individually

```bash
docker build -f Dockerfile.backend  -t rag-crewai-backend  .
docker build -f Dockerfile.frontend -t rag-crewai-frontend .
```

### What Compose sets up

- **Network**: bridge network `rag-net`. The frontend reaches the backend as `http://backend:8080`.
- **Volume**: named volume `chroma_data` mounted at `/app/backend/chroma_db` so indexed documents survive restarts.
- **Health**: the frontend waits for the backend `/health` check to pass before starting.
- **Restart policy**: `unless-stopped` on both services.

### Notes

- The backend image is large (~2 GB) because `sentence-transformers` pulls in `torch`. First build will take several minutes.
- The frontend image only contains `streamlit`, `requests`, and `python-dotenv` — it never runs the ML stack.
- To point the frontend at a backend outside Compose (e.g. a remote host), override the env var:

  ```bash
  docker compose run -e FASTAPI_BACKEND_URL=https://api.example.com frontend
  ```

---

## API Reference

### `POST /upload`

Multipart form upload. Field name: `file`. Allowed extensions: `.pdf`, `.txt`, `.md`.

Response:

```json
{
  "status": "success",
  "filename": "example.pdf",
  "chunks_indexed": 42,
  "message": "Successfully indexed example.pdf into vector store."
}
```

### `POST /query`

```json
{
  "query": "What is the main topic of the document?",
  "enable_web_search": false
}
```

Response:

```json
{ "answer": "..." }
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Programmatic Usage

You can use the RAG engine directly, without the API:

```python
import asyncio
from backend.rag_engine import ingest_document, execute_rag_crew

num_chunks = ingest_document("sample.pdf")
print(f"Ingested {num_chunks} chunks.")

answer = asyncio.run(execute_rag_crew("What is the main topic?"))
print(answer)
```

---

## Dependencies

Declared in `requirements.txt`:

- `fastapi`, `uvicorn`, `python-multipart` — backend API
- `streamlit`, `requests` — frontend
- `crewai`, `crewai-tools` — multi-agent orchestration
- `langchain`, `langchain-community`, `langchain-openai`, `langchain-huggingface` — RAG plumbing
- `chromadb` — persistent vector store
- `sentence-transformers` — local embedding model
- `pypdf` — PDF parsing
- `python-dotenv` — environment configuration

---

## Notes & Known Limitations

- **Web search toggle**: the Streamlit UI and `/query` request schema both expose `enable_web_search`, but `execute_rag_crew` in `rag_engine.py` currently ignores the flag — answers are grounded strictly to the uploaded documents. Wiring a live-search tool (e.g. DuckDuckGo) into the retrieval agent is a natural next step.
- **Persistence**: the Chroma store at `backend/chroma_db/` is not cleared between runs; re-uploading the same document will add duplicate chunks.
- **Embeddings device**: defaults to CPU. Switch `model_kwargs={"device": "cuda"}` in `rag_engine.py` if a GPU is available.
- **Ports**: backend `8080`, frontend `8501` — change via `main.py` and Streamlit CLI flags if they conflict.

---

## License

Released under the gtlabs License.

---

*Created by Tamilselvan T*
