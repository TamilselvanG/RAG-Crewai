import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag_engine import ingest_document, execute_rag_crew

app = FastAPI(title="CrewAI RAG API", version="1.0.0")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receives file, saves locally, embeds, and saves to vector store."""
    print("uploading file...")
    valid_extensions = [".txt", ".pdf", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Please upload {', '.join(valid_extensions)}"
        )

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunk_count = ingest_document(temp_path)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunk_count,
            "message": f"Successfully indexed {file.filename} into vector store."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up local file after vector ingestion
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Processes query using CrewAI agents reading from the vector store."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = await execute_rag_crew(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)