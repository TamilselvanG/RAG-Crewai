# CrewAI RAG App Backend

This project implements a Retrieval-Augmented Generation (RAG) backend using CrewAI and LangChain, enabling question answering over uploaded documents via vector search and optional live web search.

## Overview

- **Document ingestion:** Supports PDF and text files, splits documents into manageable chunks for embedding.
- **Vector store:** Uses local ChromaDB with HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`), running on CPU by default.
- **Retrieval:** Performs semantic similarity search over the embedded document chunks.
- **QA Agents:** Employs CrewAI agents with GPT-4o-mini LLM to sequentially retrieve facts and synthesize grounded answers.
- **Tools integration:** Includes tools for searching the vector database and optionally performing live DuckDuckGo web search.
- **Configurable grounding:** Toggle web search to either strictly ground answers only to uploaded document context or supplement from live internet.

## Updated Code Flow

1. **Document ingestion**: 
   Use `ingest_document(file_path: str) -> int` to load, split, and embed documents into ChromaDB.

2. **Query execution**:
   Call the async function `execute_rag_crew(user_query: str, enable_web_search: bool = False) -> str` to run CrewAI agents.

   - If `enable_web_search` is `False`, the system strictly answers only from the uploaded documents and refuses to guess.
   - If `True`, it can enrich answers using real-time web information along with uploaded docs.

3. **CrewAI Agents**:
   - **Information Retrieval Specialist**: Uses configured tools (vector DB and optionally web search) to gather factual excerpts.
   - **Knowledge Synthesizer & Writer**: Synthesizes the gathered facts into clear, concise, strictly grounded answers or refuses if no context.

4. **Tools**:
   - `Search Vector Database`: Queries the vector store with similarity search returning top relevant chunks.
   - `Internet Search`: (optional) Uses DuckDuckGo for live internet search results.

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate    # Windows
   source venv/bin/activate   # Linux/macOS
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**

   Add any API keys or secrets to `.env` if needed. Web search tool and embeddings run locally by default.

5. **Run backend functions**

   Import and call:

   ```python
   from rag_engine import ingest_document, execute_rag_crew
   import asyncio

   # Ingest document chunks into vector store
   num_chunks = ingest_document("sample.pdf")
   print(f"Ingested {num_chunks} chunks.")

   # Run a query, enabling or disabling web search
   response = asyncio.run(execute_rag_crew("What is the main topic of the document?", enable_web_search=False))
   print(response)
   ```

## Core Files

- `rag_engine.py`: Core engine for loading docs, vector database management, tool definitions, and CrewAI agent orchestration.
- `chroma_db/`: Persistent storage directory for vector embeddings.

## Dependencies

- crewai
- langchain_community
- langchain_text_splitters
- langchain_huggingface
- langchain_openai
- langchain_chroma

Refer to `requirements.txt` for full package versions.

## License

This project is released under the gtlabs.sbs License.

---

*Created by Tamilselvan T*