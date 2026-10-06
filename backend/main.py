"""
FastAPI backend for the Campus RAG Assistant.

Endpoints:
    GET  /health          - liveness check
    POST /ingest          - upload and index a notes file (PDF/TXT/MD)
    GET  /sources         - list ingested documents and chunk counts
    DELETE /sources/{name} - remove a document from the index
    POST /query           - ask a question, get a cited answer
    POST /reset           - wipe the entire vector store
"""

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_chain
from config import NOTES_DIR, OPENAI_API_KEY, RETRIEVAL_K

os.makedirs(NOTES_DIR, exist_ok=True)

app = FastAPI(title="Campus RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class QueryRequest(BaseModel):
    question: str
    k: int = RETRIEVAL_K


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_key_configured": bool(OPENAI_API_KEY),
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        num_chunks = rag_chain.ingest_file(tmp_path, source_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_indexed": num_chunks}


@app.get("/sources")
def sources():
    return {"sources": rag_chain.list_sources()}


@app.delete("/sources/{name}")
def delete_source(name: str):
    deleted = rag_chain.delete_source(name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{name}'")
    return {"source": name, "chunks_deleted": deleted}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag_chain.answer_question(request.question, k=request.k)
    return result


@app.post("/reset")
def reset():
    rag_chain.reset_store()
    return {"status": "vector store reset"}
