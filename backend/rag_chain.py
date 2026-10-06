"""
LangChain-based RAG pipeline: ingestion, ChromaDB storage, and
citation-grounded question answering.
"""

import os
import shutil
import uuid
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHAT_MODEL,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RETRIEVAL_K,
)

CITED_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a study assistant that answers questions using ONLY the student's
own notes provided below as context. Do not use outside knowledge.

Rules:
- If the answer is not contained in the context, say clearly that the notes
  don't cover this, rather than guessing.
- Every factual claim in your answer must be traceable to the context.
- After your answer, list the sources you used in a "Sources:" section,
  citing them as [filename, chunk N].

Context:
{context}

Question: {question}

Answer:"""
)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)


def _get_vectorstore() -> Chroma:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
    )


def _load_document(file_path: str) -> list[Document]:
    """Load a single file (PDF, TXT, or MD) into LangChain Documents."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader.load()


def ingest_file(file_path: str, source_name: str | None = None) -> int:
    """
    Load, chunk, embed, and store a single file's contents in ChromaDB.
    Returns the number of chunks stored.
    """
    source_name = source_name or os.path.basename(file_path)

    raw_docs = _load_document(file_path)
    if not raw_docs or not any(d.page_content.strip() for d in raw_docs):
        raise ValueError(
            f"No extractable text found in {source_name}. "
            f"It may be empty or a scanned/image-only PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # Tag each chunk with source filename + chunk index for citations
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_uid"] = str(uuid.uuid4())

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)


def delete_source(source_name: str) -> int:
    """Remove all chunks belonging to a given source file from the store."""
    vectorstore = _get_vectorstore()
    existing = vectorstore.get(where={"source": source_name})
    ids = existing.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def list_sources() -> list[dict]:
    """Return a summary of ingested source documents and chunk counts."""
    vectorstore = _get_vectorstore()
    data = vectorstore.get()
    metadatas = data.get("metadatas", [])
    counts: dict[str, int] = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": src, "chunks": n} for src, n in counts.items()]


def reset_store() -> None:
    """Wipe the entire vector store (useful for demos / evals)."""
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)


def retrieve(question: str, k: int = RETRIEVAL_K) -> list[Document]:
    """Return the top-k most relevant chunks for a question."""
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def retrieve_with_scores(question: str, k: int = RETRIEVAL_K) -> list[tuple]:
    """Return top-k chunks with similarity scores (lower distance = more relevant)."""
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search_with_score(question, k=k)


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        idx = doc.metadata.get("chunk_index", "?")
        parts.append(f"[{src}, chunk {idx}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, k: int = RETRIEVAL_K) -> dict:
    """
    Full RAG call: retrieve relevant chunks, generate a cited answer.
    Returns a dict with the answer text and the source chunks used.
    """
    docs = retrieve(question, k=k)
    if not docs:
        return {
            "answer": "No notes have been ingested yet, so I can't answer this.",
            "sources": [],
        }

    context = _format_context(docs)
    llm = _get_llm()
    chain = CITED_ANSWER_PROMPT | llm

    response = chain.invoke({"context": context, "question": question})

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "excerpt": doc.page_content[:300],
        }
        for doc in docs
    ]

    return {"answer": response.content, "sources": sources}
