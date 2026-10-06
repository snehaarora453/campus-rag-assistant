# Campus RAG Assistant

A RAG system that answers questions from a student's notes with **cited,
grounded answers** instead of generic LLM responses — built with FastAPI
(backend), Streamlit (frontend), LangChain (orchestration), ChromaDB
(vector store), and the OpenAI API (embeddings + generation).

Retrieval accuracy is measured against a labeled test set using Hit@K and
Mean Reciprocal Rank (MRR).

## Architecture

```
campus-rag/
├── backend/
│   ├── main.py          # FastAPI app: /ingest, /query, /sources, /reset
│   ├── rag_chain.py      # LangChain logic: chunking, embeddings, ChromaDB, cited QA
│   ├── config.py         # Central settings
│   └── .env.example
├── frontend/
│   └── streamlit_app.py  # Chat UI, calls the FastAPI backend over HTTP
├── eval/
│   ├── test_set.json     # Labeled QA pairs (question -> expected source doc)
│   ├── evaluate.py        # Computes Hit@K and MRR against the test set
│   └── ingest_notes.py    # Bulk-loads data/notes/ directly, no API needed
├── data/
│   ├── notes/             # Sample notes (swap in your own)
│   └── chroma_db/         # ChromaDB persistent storage (created on first run)
└── requirements.txt
```

The backend and frontend are **separate processes** communicating over HTTP
(`localhost:8000`), matching a real client/server RAG deployment rather than
a single monolithic script.

## 1. Requirements

- Python 3.10 or 3.11 (3.12 also tested and works with this requirements.txt)
- An OpenAI API key: https://platform.openai.com/api-keys

## 2. Setup

**Always use a fresh virtual environment** — installing into your global
Python is the most common cause of "it doesn't work" package conflicts.

```bash
cd campus-rag
python3 -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Two dependency bugs this project pre-emptively fixes

This exact stack (LangChain + ChromaDB + OpenAI SDK) has real, current
version-compatibility landmines if you `pip install` the packages without
pinning. `requirements.txt` already pins around both, but it's worth knowing
what they are in case you add/upgrade packages later:

1. **`httpx==0.27.2` pin** — `openai`'s HTTP client wrapper (as of SDK 1.54.x)
   still passes a `proxies` argument that was removed in `httpx==0.28`. Left
   unpinned, pip installs the newest httpx and every OpenAI API call crashes
   at client-init time with `TypeError: Client.__init__() got an unexpected
   keyword argument 'proxies'`.
2. **`posthog==3.7.0` pin** — ChromaDB's built-in (and non-optional) usage
   telemetry calls `posthog.capture()` using an old positional-argument
   signature. The latest `posthog` package rewrote that method's signature,
   so unpinned installs spam `Failed to send telemetry event... capture()
   takes 1 positional argument but 3 were given` on every single vector
   store operation. Harmless, but noisy enough to look like something's
   broken.

Both were caught by actually installing fresh and exercising the code before
handing this back to you, not just written from memory — so if the errors
above look familiar, that's likely what bit you in the first version.

## 3. Add your API key

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set `OPENAI_API_KEY`.

## 4. Run it

**Terminal 1 — backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The sidebar
will show whether it can reach the backend and whether an API key is
configured.

## 5. Use it

1. Upload PDF, TXT, or Markdown notes (or just use the 3 sample `.md` files
   already in `data/notes/` — see step 6 to load them without the UI).
2. Click **Process documents**.
3. Ask questions in the chat box. Every answer cites which file and chunk it
   came from — expand **Sources used** under any answer to check it.
4. Remove a document from the sidebar's knowledge-base list, or wipe
   everything with **Reset entire knowledge base**.

## 6. Evaluate retrieval accuracy

The labeled test set in `eval/test_set.json` has 3 sample questions matched
to the 3 sample notes files in `data/notes/`. To reproduce the resume-line
claim of "evaluated retrieval accuracy on a labeled test set":

```bash
# Load the sample notes directly into ChromaDB (bypasses the API/UI)
cd eval
python ingest_notes.py

# Run the evaluation
python evaluate.py
python evaluate.py --k 5    # try a different top-K
```

This prints **Hit@K** (fraction of questions where the correct source
appeared in the top-K retrieved chunks) and **MRR** (Mean Reciprocal Rank —
rewards the correct source appearing higher, not just present) and writes
`eval/eval_report.json` with per-question detail.

**To evaluate your own notes:** replace the files in `data/notes/`, then
rewrite `eval/test_set.json` with questions whose answers live in your real
notes and the correct `expected_source` filename for each.

## API reference (backend)

| Method | Path              | Description                          |
|--------|-------------------|---------------------------------------|
| GET    | `/health`         | Liveness + API key configured check   |
| POST   | `/ingest`         | Upload a file (multipart) to index    |
| GET    | `/sources`        | List indexed documents + chunk counts |
| DELETE | `/sources/{name}` | Remove a document from the index      |
| POST   | `/query`          | `{"question": "...", "k": 4}` → cited answer |
| POST   | `/reset`          | Wipe the entire vector store          |

Interactive API docs are auto-generated by FastAPI at
`http://localhost:8000/docs` once the backend is running.

## Notes / limitations

- Scanned/image-only PDFs won't extract text (no OCR) — `/ingest` returns a
  422 error naming the file.
- ChromaDB persists to `data/chroma_db/` on disk, so your indexed notes
  survive a restart (unlike the in-memory version of this project).
- Costs: embeddings and chat calls use your OpenAI API key and incur small
  per-token charges.
- If `pip install` still fails after using a fresh venv, run `python
  --version` — anything 3.13+ may have compiled-wheel gaps for some pinned
  packages; use 3.10/3.11 and recreate the venv.
