# CrewAI RAG App Backend

This project implements a Retrieval-Augmented Generation (RAG) backend using CrewAI and LangChain, enabling question answering over uploaded documents via vector search.

## Overview

- **Document ingestion:** Supports PDF and text files, splitting documents into chunks for embedding.
- **Vector store:** Uses local ChromaDB with HuggingFace embeddings (`all-MiniLM-L6-v2`) on CPU by default.
- **Retrieval:** Performs similarity search over embedded chunks.
- **QA Agents:** CrewAI agents run sequential tasks for retrieval and answer synthesis leveraging GPT-4o-mini LLM.
- **Tools integration:** Custom CrewAI tool for vector database search.

## Features

- Ingest documents and embed them into a vector store for retrieval.
- Search for relevant document chunks using semantic similarity.
- Generate clear, factual answers grounded in retrieved context.
- Modular architecture using CrewAI agents and tasks.

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

4. **Set up environment variables**

   Ensure any API keys needed by underlying libraries (e.g., OpenAI) are set. This project primarily uses local models.

5. **Run the backend**

   Integrate with your server or call functions like `ingest_document()` and `execute_rag_crew()` asynchronously.

## Core Files

- `rag_engine.py`: Main engine for ingesting docs, managing vector store, and running CrewAI agents.
- `chroma_db/`: Directory where vector embeddings and database persist.

## Usage Example

```python
from rag_engine import ingest_document, execute_rag_crew
import asyncio

# Ingest a new document
num_chunks = ingest_document("sample.pdf")
print(f"Ingested {num_chunks} chunks.")

# Run a query
response = asyncio.run(execute_rag_crew("What is the main topic of the document?"))
print(response)
```

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
