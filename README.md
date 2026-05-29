# Multimodal RAG Assistant

A fully **local**, **free**, end-to-end Retrieval-Augmented Generation system that understands PDFs, Word documents, images, audio, video, source code, and **web pages** — and lets you chat with all of it through a polished **Next.js + Tailwind** interface backed by a **FastAPI** Python server.

No OpenAI API keys. No cloud services. Everything runs on your machine via [Ollama](https://ollama.com).

---

##  Features

- **Multimodal ingestion** — PDFs, DOCX, images, MP3/WAV audio, MP4/MKV video, and 20+ code formats
- **Deterministic Markdown chunking** — structure-aware splitter that respects headings, fenced code, tables, and blockquotes; ~200× faster than the legacy LLM chunker and immune to JSON-parse failures on long docs. The LLM chunker is still available via `chunker_strategy="llm"`.
- **Contextual Retrieval** — every chunk is enriched with a situating context sentence before embedding, reducing retrieval failures by ~49% ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval))
- **Hybrid retrieval** — dense vector search (ChromaDB + nomic-embed-text) fused with BM25 sparse search via Reciprocal Rank Fusion (RRF)
- **Optional cross-encoder reranking** — a second-stage `BAAI/bge-reranker-v2-m3` re-scores fused candidates for relevance, typically lifting top-1 precision by ~5–15%
- **Multi-turn conversational retrieval** — follow-up questions with pronouns ("how does that compare to Q2?") are rewritten into self-contained search queries against the chat history before retrieval
- **Cross-file numbered citations** — answers cite each claim with `[1]`, `[2]`, … pointing at a numbered source list, so attribution stays clear even when the response stitches together facts from multiple files
- **Metadata filtering in the UI** — restrict retrieval by **modality** (Document / Audio / Image / Video / Code), **source file**, or **indexed-date range** directly from the sidebar
- **Web URL ingestion** — paste any article URL; `trafilatura` extracts clean main-content Markdown which flows through the same chunking → contextualisation → indexing pipeline as files
- **Persistent memory & prompts** — a user-controlled system prompt + a list of "standing facts" notes that get injected into every answer (no re-indexing needed)
- **Fully local LLMs** — Llama 3 for chunking, context generation, and answering; LLaVA for image understanding; Whisper for transcription
- **Tabbed Next.js + Tailwind UI** — Chat / Knowledge Base / Memory tabs against a FastAPI backend. A legacy single-process Streamlit UI is also kept as a no-Node-required fallback.
- **Persistent indexes** — ChromaDB and BM25 survive restarts; re-indexing a file automatically deduplicates

---

## Architecture

```
                      ┌───────────────────────────────────────┐
                      │   Next.js + Tailwind UI (port 3000)   │
                      │   Chat · Knowledge Base · Memory      │
                      └───────────────────┬───────────────────┘
                                          │  /api/* (proxied)
                      ┌───────────────────▼───────────────────┐
                      │      FastAPI backend (port 8000)      │
                      │       api.py — single shared server    │
                      └───────┬───────────────────────┬───────┘
                              │                       │
        ╔═════════════════════╧══════════╗   ╔════════╧════════════╗
        ║         INGESTION              ║   ║       QUERY          ║
        ╠════════════════════════════════╣   ╠══════════════════════╣
   File ║                                ║   ║  Question            ║
   /URL ║  DataRouter / URLProcessor     ║   ║      │               ║
        ║      │                         ║   ║      ▼               ║
        ║      ├─► DocumentProcessor     ║   ║  QueryRewriter       ║
        ║      ├─► AudioProcessor        ║   ║  (multi-turn → self- ║
        ║      ├─► ImageProcessor        ║   ║   contained query)   ║
        ║      ├─► VideoProcessor        ║   ║      │               ║
        ║      ├─► CodeProcessor         ║   ║      ▼               ║
        ║      └─► URLProcessor          ║   ║  Embed + ChromaDB    ║
        ║              │                 ║   ║      +               ║
        ║              ▼                 ║   ║  BM25 sparse         ║
        ║      MarkdownUnifier           ║   ║      │               ║
        ║              │                 ║   ║      ▼               ║
        ║              ▼                 ║   ║  HybridRetriever     ║
        ║      MarkdownChunker (default) ║   ║  (RRF fusion, fills  ║
        ║      | ChunkingAgent (opt-in)  ║   ║   sparse-only hits)  ║
        ║              │                 ║   ║      │               ║
        ║              ▼                 ║   ║      ▼               ║
        ║      ContextualRetriever       ║   ║  CrossEncoderReranker║
        ║      (situating sentence       ║   ║  (optional)          ║
        ║       prepended per chunk)     ║   ║      │               ║
        ║              │                 ║   ║      ▼               ║
        ║              ▼                 ║   ║  Llama 3 generation  ║
        ║      Embed (nomic-embed-text)  ║   ║  + memory + custom   ║
        ║              │                 ║   ║    system prompt     ║
        ║              ▼                 ║   ║      │               ║
        ║      ChromaDB + BM25Index      ║   ║      ▼               ║
        ║                                ║   ║  Answer + [1][2] cites║
        ╚════════════════════════════════╝   ╚══════════════════════╝
                              │
                      ┌───────▼───────┐
                      │  memory.json  │  (custom system prompt
                      │               │   + user memory notes)
                      └───────────────┘
```

> Query over-retrieves `top_k × 2` at the fusion stage and again at the reranker, so the final `top_k` shown to the LLM is the cream of a much larger candidate pool. When a metadata filter is active, the BM25 pool is doubled again to compensate for filtering loss.

---

## 🚀 Running the App

The app has three long-running processes:

| Process | What it does | Port |
|---------|--------------|------|
| **Ollama** | Serves the local LLMs (Llama 3, LLaVA, nomic-embed-text) | 11434 |
| **FastAPI backend** (`api.py`) | RAG pipeline, ingestion, chat, memory | 8000 |
| **Next.js frontend** (`frontend/`) | The UI you actually use in the browser | 3000 |

The frontend proxies `/api/*` to the backend, so you only ever open one URL: **http://localhost:3000**.

### One-time setup

#### 1. System tools

You need **Python 3.10+**, **Node.js 18+**, and **FFmpeg** (for audio/video):

```bash
# macOS
brew install python node ffmpeg

# Ubuntu / Debian
sudo apt install python3 python3-venv nodejs npm ffmpeg
```

Install **Ollama** from <https://ollama.com/download>.

#### 2. Pull the local models

```bash
ollama pull llama3            # chunking, context, query rewriting, answering
ollama pull llava             # image understanding (~4 GB)
ollama pull nomic-embed-text  # embeddings (~274 MB)
```

#### 3. Python virtual environment + deps

```bash
cd /path/to/Advanced_RAG
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Whisper and Docling models are downloaded automatically on first use. The optional reranker (`BAAI/bge-reranker-v2-m3`, ~1.1 GB) downloads only when you set `RAG_USE_RERANKER=true`.

#### 4. Install the frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Every-time you run it

You need three terminals. The first two run forever; the third is where you start the actual app.

```bash
# Terminal 1 — Ollama LLM server (must be running)
ollama serve

# Terminal 2 — FastAPI backend
cd /path/to/Advanced_RAG
source venv/bin/activate
uvicorn api:app --reload --port 8000

# Terminal 3 — Next.js frontend
cd /path/to/Advanced_RAG/frontend
npm run dev
```

Then open **<http://localhost:3000>** in your browser. You should see the dark-themed UI with three tabs (Chat / Knowledge Base / Memory & Prompts) and a live chunk count in the header. If the header says **"backend offline"**, the FastAPI server (terminal 2) isn't reachable.

### Enabling the optional cross-encoder reranker

The reranker is off by default. To turn it on, install the extra dep and set an environment variable before launching the backend:

```bash
pip install sentence-transformers
RAG_USE_RERANKER=true uvicorn api:app --reload --port 8000
```

Adds ~100–300 ms per query, typically lifts top-1 precision by 5–15%.

### Production build (frontend)

If you'd rather run the optimised frontend instead of the dev server:

```bash
cd frontend
npm run build && npm start
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Header says "backend offline" | FastAPI isn't running or port 8000 is busy | Start terminal 2; or run on another port and update `frontend/next.config.js` |
| "Cannot reach Ollama server" on startup | `ollama serve` not running | Start terminal 1 |
| URL ingest hangs ≥ 30 s then errors | Next dev's proxy timeout while the backend keeps working | The page is still ingested — refresh the KB tab to see it. The backend uses Ollama which is single-threaded for long contextual-retrieval passes. |
| `npm install` errors about Node version | Node < 18 | Upgrade Node |
| Ingest crashes on a long document | Was the LLM chunker; not the default anymore | Make sure `chunker_strategy="markdown"` (the default) — only an issue if you explicitly opted into `"llm"` |
| Reranker import error | `sentence-transformers` not installed | `pip install sentence-transformers` |

### Legacy Streamlit UI (alternative)

The original single-process Streamlit UI still works if you don't want to install Node:

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>. It has all the same retrieval features but a simpler look and no separate Memory tab.

---

## 📖 Using the App

The UI at <http://localhost:3000> has three tabs.

### 💬 Chat tab

- Type a question, hit Enter, get an answer with `[1]`, `[2]` … citations that link to a sources panel under the response.
- Follow-up questions with pronouns ("how does that compare to Q2?") are automatically rewritten into self-contained queries using the chat history before retrieval.
- A collapsible **Filters** strip lets you scope retrieval by **modality** (Document / Audio / …) or **source file**. Active filters appear as removable chips.
- The header shows the live count of total chunks and sources.

### 📚 Knowledge Base tab

- **Upload a file** — drag-and-drop a PDF, DOCX, image, audio, video, or code file.
- **Ingest a web URL** — paste any article URL; the page is fetched, cleaned, chunked, contextualised, embedded, and indexed.
- A table lists every indexed source with filename/title, modality, chunk count, indexed date, and a delete button.

### 🧠 Memory & Prompts tab

Two persistent, user-controlled prompt additions:

- **Custom system instructions** — e.g. "answer concisely in British English; define acronyms on first use."
- **Memory notes** — standing facts the model should always respect, like "Acme has been our client since 2019" or "the fiscal year ends in March."

Both are injected into the system prompt on every answer with **no re-indexing required**. Memory notes are explicitly marked in the prompt so the LLM doesn't confuse them with source chunks and never cites them as `[N]`.

---

## 🛠️ CLI Usage (no UI)

You can drive ingestion and retrieval entirely from the command line — handy for scripting or batch indexing.

```bash
# Index a file end-to-end
python rag_connector.py index quarterly_report.pdf

# Query the knowledge base
python rag_connector.py query "What were the main revenue drivers in Q3?"

# Query with modality filter
python rag_connector.py query "Summarise the meeting" --modality Audio

# View index stats
python rag_connector.py stats

# Remove a source
python rag_connector.py delete quarterly_report.pdf
```

You can also run the ingestion pipeline directly and dump chunks as JSON without writing to the index:

```bash
# Print chunks to stdout
python multimodal_rag_pipeline.py report.pdf

# Save chunks to a file
python multimodal_rag_pipeline.py meeting.mp3 chunks.json

# Skip contextual retrieval (faster, no LLM passes per chunk)
python multimodal_rag_pipeline.py diagram.png out.json --no-context
```

---

## 📁 Project Structure

```
.
├── api.py                      # FastAPI backend — chat, ingest, sources, memory
├── memory_store.py             # JSON-backed system prompt + memory notes
├── rag_connector.py            # Core façade: index(), index_url(), query()
│   ├── OllamaEmbedder          # nomic-embed-text dense embeddings
│   ├── BM25Index               # Sparse keyword index (bm25_index.json)
│   ├── ChromaStore             # Dense vectors + metadata (./chroma_db/)
│   ├── CrossEncoderReranker    # Optional bge-reranker-v2-m3
│   ├── HybridRetriever         # RRF fusion
│   ├── QueryRewriter           # Multi-turn conversational rewriting
│   └── RAGConnector            # Top-level orchestrator
├── multimodal_rag_pipeline.py  # Ingestion pipeline
│   ├── DataRouter              # File-type dispatcher
│   ├── DocumentProcessor       # PDF/DOCX → Markdown via Docling
│   ├── AudioProcessor          # Audio → transcript via Whisper
│   ├── ImageProcessor          # Image → description via LLaVA
│   ├── VideoProcessor          # Video → audio + keyframes
│   ├── CodeProcessor           # Source code → fenced Markdown
│   ├── URLProcessor            # Web page → clean Markdown via trafilatura
│   ├── MarkdownUnifier         # Attaches metadata header
│   ├── MarkdownChunker         # Deterministic structure-aware splitter (default)
│   ├── ChunkingAgent           # Legacy LLM chunker (opt-in: chunker_strategy="llm")
│   ├── ContextualRetriever     # Parallel context enrichment via Llama 3
│   └── MultimodalRAGPipeline   # High-level orchestrator (ingest + ingest_url)
├── app.py                      # Legacy Streamlit UI (still works)
├── frontend/                   # Next.js 14 + TypeScript + Tailwind UI
│   ├── app/page.tsx            # 3-tab shell
│   ├── app/components/         # ChatTab · KnowledgeBaseTab · MemoryTab
│   ├── lib/api.ts              # Typed client for the FastAPI backend
│   └── package.json
├── bm25_index.json             # Persisted BM25 corpus (auto-created)
├── chroma_db/                  # ChromaDB storage (auto-created)
├── memory.json                 # User memory state (auto-created)
└── requirements.txt
```

---

## ⚙️ Configuration

All components accept constructor parameters for customisation. Key defaults:

**RAGConnector + MultimodalRAGPipeline:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contextual_retrieval` | `True` | Enable/disable Anthropic-style context enrichment |
| `context_max_workers` | `4` | Parallel LLM calls for context generation |
| `chunker_strategy` | `"markdown"` | `"markdown"` for the deterministic chunker (recommended) or `"llm"` for the legacy ChunkingAgent |
| `embed_model` | `nomic-embed-text` | Ollama embedding model |
| `use_reranker` | `False` | Enable cross-encoder reranking after RRF fusion |
| `reranker_model` | `BAAI/bge-reranker-v2-m3` | sentence-transformers cross-encoder model |
| `generation_model` | `llama3` | Ollama model for final answer generation |
| `keyframe_interval` | `30` | Seconds between video keyframe samples |
| `whisper_model` | `base` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`) |

**Backend environment variables (read by `api.py`):**

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_USE_RERANKER` | `false` | Set to `true` to enable the optional cross-encoder reranker globally |

Example — enable reranking and use more parallelism for context generation:

```python
from rag_connector import RAGConnector

rag = RAGConnector(
    contextual_retrieval=True,
    context_max_workers=8,   # more parallelism on GPU
    use_reranker=True,       # cross-encoder second-stage ranking
)
rag.index("lecture.mp4")

results = rag.query("What were the key takeaways?", top_k=5)
print(results["answer"])
```

---

## ✂️ How the Markdown Chunker Works

Documents are split into chunks by a deterministic structure-aware walker:

1. **Tokenize** the unified Markdown into structural blocks: headings (h1–h6), paragraphs, fenced code blocks, tables, and blockquotes.
2. **Pack** blocks greedily up to ~1,500 chars per chunk, preferring heading boundaries as split points whenever the buffer is at least half-full.
3. **Preserve atomicity** — code blocks, tables, and blockquotes are never split across chunks (the same rule the LLM chunker tried to follow but occasionally violated).
4. **Long paragraphs** that exceed the max are split at sentence boundaries instead of mid-word.
5. **Outline prepending** — chunks that begin mid-section get the parent heading breadcrumb prepended so they remain self-contained when retrieved out of order. Chunks that already start with a heading need no prefix.
6. **Tiny tail merge** — a final chunk under MIN_CHARS is merged back into its predecessor rather than emitted standalone.

This replaced the prior `ChunkingAgent` (LLM with a JSON-emit prompt) as the default. The LLM chunker silently truncated past `num_ctx=8192` and occasionally returned malformed JSON on long inputs, killing ingest. The deterministic chunker runs in ~100 ms on the same document, never crashes, and emits chunks that are exact substrings of the source (no paraphrasing drift). Both are still available — pick with `MultimodalRAGPipeline(chunker_strategy="markdown" | "llm")`.

---

## 🔬 How Contextual Retrieval Works

Naive chunking strips the surrounding context that makes a chunk interpretable. A chunk reading *"Revenue declined 12% quarter-over-quarter"* is ambiguous in isolation — which company? which quarter?

Contextual Retrieval ([Anthropic, Sept 2024](https://www.anthropic.com/news/contextual-retrieval)) fixes this by passing the **whole document** to the LLM alongside each chunk and asking it to write a 1–2 sentence situating context:

> *"This chunk is from the Acme Corp Q3 2024 earnings report, in the North American retail segment section."*

That context is prepended to the chunk before embedding **and** BM25 indexing. Result: ~49% fewer retrieval failures on Anthropic's benchmarks.

In this project the contextualisation calls run in parallel via `ThreadPoolExecutor`, keeping latency manageable even for large documents.

---

## 🗂️ Metadata Filtering & Cross-File Citations

Every chunk is stored with structured metadata: `source` (filename), `modality`, `summary`, `keywords`, `context`, `indexed_at` (ISO string), and `indexed_ts` (Unix int for range queries).

**Filtering** happens at query time. The Chat tab exposes a collapsible filter bar with multi-select dropdowns; the API also accepts a date-range filter. All active filters are combined into a single ChromaDB `where` clause and applied to both retrieval stages:

| Filter | Backend behaviour |
|--------|-------------------|
| **Modality** (multi-select) | `{"modality": {"$in": [...]}}` — dense retrieval filters natively; BM25 results are post-filtered via `ChromaStore.filter_ids` since rank_bm25 is metadata-blind. |
| **Source file** (multi-select) | `{"source": {"$in": [...]}}` — same path. |
| **Indexed date range** | `{"indexed_ts": {"$gte": …, "$lte": …}}` — exact range query over the integer timestamp stored at ingest. |

When more than one constraint is active they are combined with `$and`. Sparse retrieval over-retrieves (`top_k × 4`) when a filter is active so the surviving set is still rich enough for RRF + reranking.

**Citations** were upgraded for multi-file scenarios. The LLM no longer writes `[Source: filename]` inline; instead it cites each claim with `[1]`, `[2]`, … matching a numbered source list rendered in the UI. The system prompt explicitly tells the model to **not blend facts across files** — every numbered claim must come from the chunk with that number. The "Sources used" panel shows how many chunks came from how many distinct files, with a short snippet of each chunk so attribution is verifiable.

---
## 🎯 How Reranking Works

RRF fusion ranks results by their *position* in the dense and sparse result lists — it never directly measures how well a chunk answers the query. This can promote chunks that merely share keywords over chunks that actually contain the answer.

The optional **cross-encoder reranker** fixes this. Unlike the bi-encoder used for embeddings (which encodes query and chunk separately), a cross-encoder feeds the `(query, chunk)` pair through the model *together*, producing a single relevance score that captures their interaction.

```
Query: "What caused the revenue decline?"

  RRF order              →  Reranked order
  1. "Revenue fell 12%"      1. "Cause: market saturation"   ← actually answers it
  2. "Cause: market sat."    2. "Revenue fell 12%"
```

The query pipeline over-retrieves `top_k × 2` candidates at the fusion stage, the reranker scores them all, and only the best `top_k` reach the LLM. Cost is ~100–300 ms for a typical candidate set — negligible next to generation — and it degrades gracefully (falls back to RRF order) if the model fails to load.

Enable it with `RAGConnector(use_reranker=True)` after installing `sentence-transformers`.

---

## Supported Sources

| Category | Inputs |
|----------|--------|
| Documents | `.pdf`, `.docx`, `.doc`, `.odt`, `.pptx`, `.xlsx` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.opus` |
| Images | `.jpg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp` |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v` |
| Code | `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.rb`, `.sh`, `.sql`, `.html`, `.css`, `.yaml`, `.json`, and more |
| **Web** | Any `http://` or `https://` URL — fetched via `httpx`, cleaned by `trafilatura` |

---

## Dependencies

**Python (`requirements.txt`):**

```
# Web framework — backend
fastapi
uvicorn[standard]
python-multipart       # multipart file upload support
streamlit              # legacy single-process UI (still works)

# LLM / RAG
ollama                 # Local LLM + embedding inference
chromadb               # Vector database
rank_bm25              # Sparse BM25 retrieval
sentence-transformers  # Cross-encoder reranker (optional)

# Ingestion
httpx                  # Web URL fetching
trafilatura            # Web page → clean Markdown
docling                # PDF / DOCX → Markdown
openai-whisper         # Local speech-to-text
ffmpeg-python          # Video processing bindings
pydantic               # Data validation (used by FastAPI)
```

**Frontend (`frontend/package.json`):**

```
next ^14.2          react ^18.3      react-dom ^18.3
lucide-react        react-markdown   remark-gfm
tailwindcss ^3.4    typescript ^5.5
```

---

## 🗺️ Roadmap

- [x] Re-ranking with a cross-encoder model
- [x] Multi-turn / conversational retrieval (query rewriting from chat history)
- [x] Multi-document cross-file citation
- [x] Metadata filtering in the UI (by modality, date, source)
- [x] Support for web URL ingestion
- [x] Full UI overhaul (FastAPI + Next.js + Tailwind, 3-tab layout)
- [x] Persistent memory & custom prompts
- [ ] Docker Compose setup for one-command deployment
- [ ] Page-number / timestamp granularity in citations (PDF pages, video timestamps)
- [ ] Streaming token-by-token chat responses

---

## 👤 Author

**Abdelrahman Ezz-Eldin**  
AI Engineering Enthusiast | Computer Engineering Student  
[LinkedIn](https://linkedin.com/in/abdelrahman-ezz-44011a392) · Alexandria, Egypt
