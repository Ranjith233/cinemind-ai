# 🎬 CineMind AI – Advanced Cinema Knowledge Assistant

**CineMind AI** is a production-oriented, modular, and extensible Retrieval-Augmented Generation (RAG) platform purpose-built for cinema scholarship, screenplay analysis, and film research.

Unlike rudimentary PDF chatbots, CineMind AI features an enterprise multi-stage retrieval architecture combining **Dense Semantic Vector Search**, **Sparse BM25 Lexical Retrieval**, **Reciprocal Rank Fusion (RRF)**, and **Deep Cross-Encoder Reranking**, delivering grounded answers with page-level citations.

The primary knowledge foundation is the book **_Quentin Tarantino_ by Ian Nathan** (277 pages), fully indexed and queryable across Tarantino's filmmaking journey, screenwriting techniques, nonlinear narrative structures, cinephilic influences (Video Archives, Godard, Leone, De Palma), and complete filmography.

---

## 🏛️ Advanced Architecture Pipeline

```
                                      DOCUMENT INGESTION PIPELINE
[ PDF / DOCX / TXT / CSV ]
           │
           ▼
[ PyMuPDF Text Sanitization & Chapter Detection ]
           │
           ▼
[ Tokenizer-Aware Paragraph Chunking (800 tok / 120 ovlp) ]
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                              ▼                              ▼
[ SHA-256 Hash Duplicate Guard ]   [ Sentence Transformers Embed ]   [ SQLite Document Registry ]
           │                              │                              │
           └──────────────────────────────┼──────────────────────────────┘
                                          ▼
                         [ Persistent ChromaDB Storage ]
                                          │
                                          ▼
                            [ In-Memory BM25 Lexical Index ]

────────────────────────────────────────────────────────────────────────────────────────────

                                      HYBRID RETRIEVAL & GENERATION
[ User Inquiry ] ──► [ Conversational Context Reformulation (Gemini) ]
                                          │
                                          ▼
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
     [ Dense Vector Similarity ]                       [ Sparse BM25 Keyword Search ]
     (Cosine Metric, k=20)                             (Okapi BM25, k=20)
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          ▼
                         [ Reciprocal Rank Fusion (RRF, k=60) ]
                                          │
                                          ▼
                    [ Cross-Encoder Deep Reranker (ms-marco-MiniLM) ]
                                          │
                                          ▼ (Top 4 Chunks Selected)
                    [ Grounded Prompt Assembly & Evidence Boundary ]
                                          │
                                          ▼
                    [ Google Gemini LLM (Streaming Inference) ]
                                          │
                                          ▼
              [ Markdown Answer + Page Citations + Supporting Excerpts ]
```

---

## 🌟 Key Features

- **Multi-Stage Hybrid Retrieval**: Combines semantic embeddings with BM25 lexical keyword matching using Reciprocal Rank Fusion ($k=60$).
- **Deep Cross-Attention Reranking**: Re-scores candidate passages with `cross-encoder/ms-marco-MiniLM-L-6-v2` to eliminate semantic drift.
- **Context-Aware Query Rewriting**: Resolves conversational pronouns and context across multi-turn dialogues into standalone search queries.
- **Strict Grounding & Citation Integrity**: Answers cite exact document names, chapters, and page numbers with verbatim supporting excerpts.
- **Tokenizer-Aware Chunking**: Preserves paragraph and sentence boundaries with configurable token limits (default 800 tokens, 120 overlap).
- **Persistent Storage**: Retains vector indices across reboots in ChromaDB and manages document metadata and chat sessions in SQLite.
- **Duplicate Prevention**: SHA-256 cryptographic hashing prevents redundant document ingestion while enabling single-click re-indexing.
- **Multi-Format Ingestion**: Ingests PDF books, DOCX scripts, TXT analyses, and CSV metadata.
- **Streaming UI**: Real-time token streaming powered by the official `google-genai` SDK.
- **Premium Cinematic UI/UX**: Custom dark charcoal and navy glassmorphic theme with gold accents, responsive cards, and diagnostics.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Runtime** | Python 3.11 | High-performance isolated environment |
| **Frontend** | Streamlit | Cinematic dark glassmorphic interface |
| **LLM Inference** | Google Gemini (`google-genai` SDK) | `gemini-2.5-flash` / `gemini-1.5-flash` |
| **Dense Embeddings** | Sentence Transformers | `all-MiniLM-L6-v2` (384-dimensional) |
| **Sparse Retrieval** | Rank-BM25 | BM25Okapi lexical token index |
| **Reranker** | Sentence Transformers Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector Database** | ChromaDB | Persistent local vector store (`hnsw:space: cosine`) |
| **Document Parser** | PyMuPDF (`pymupdf`) | High-speed PDF text, TOC, and pagination extraction |
| **Relational DB** | SQLite | Document metadata, hash registry, and chat history |
| **Evaluation** | Automated Evaluation Framework / Ragas | Precision, Recall, MRR, Citation Correctness |
| **Testing** | Pytest | Comprehensive test suite covering all modules |

---

## 📁 Project Structure

```
cinemind-ai/
├── app.py                      # Main Streamlit application entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & benchmark report
├── .env.example                # Environment configuration template
├── .gitignore                  # Git tracking exclusion rules
│
├── config/
│   ├── settings.py             # Global settings, paths, and hyperparameters
│   └── logger.py               # Centralized logging configuration
│
├── core/
│   ├── chapter_detector.py     # PDF TOC and regex chapter detector
│   ├── pdf_loader.py           # PyMuPDF extractor with text dehyphenation
│   ├── document_loader.py      # Unified loader for PDF, DOCX, TXT, CSV
│   ├── text_splitter.py        # Tokenizer-aware paragraph boundary chunker
│   ├── embeddings.py           # Modular SentenceTransformer embeddings
│   ├── vector_store.py         # Persistent ChromaDB collection manager
│   ├── keyword_search.py       # BM25Okapi lexical retrieval engine
│   ├── retriever.py            # Hybrid retriever with Reciprocal Rank Fusion
│   ├── reranker.py             # Cross-Encoder deep relevance reranker
│   ├── llm.py                 # Google Gemini client via google-genai SDK
│   ├── prompts.py              # Cinema specialist system prompts & templates
│   └── rag_pipeline.py         # End-to-end RAG orchestrator
│
├── database/
│   ├── metadata.py             # SQLite document registry & hash tracking
│   └── conversation.py         # SQLite chat sessions and message history
│
├── evaluation/
│   ├── evaluate.py             # Automated empirical evaluation runner
│   ├── questions.json          # Ground-truth cinema benchmark questions
│   └── report.json             # Measured evaluation output
│
├── ui/
│   ├── components.py           # Cinematic CSS theme & branding headers
│   ├── sidebar.py              # Session management, upload, model settings
│   ├── chat.py                 # Interactive chat stream & suggested prompts
│   ├── library.py              # Document management, stats, re-index controls
│   └── sources.py              # Collapsible citation & excerpt drawer
│
├── scripts/
│   └── ingest_book.py          # Script to batch-ingest Quentin-Tarantino.pdf
│
├── tests/
│   ├── test_pdf_loader.py      # PDF extraction and chunking tests
│   ├── test_embeddings.py      # Embedding generation and ChromaDB tests
│   ├── test_retriever.py       # BM25, hybrid RRF, and reranker tests
│   └── test_rag.py             # End-to-end RAG integration tests
│
├── data/
│   ├── source/                 # Primary knowledge source (Quentin-Tarantino.pdf)
│   ├── uploads/                # User-uploaded cinema files
│   └── processed/              # Processed artifacts
│
├── storage/
│   ├── chroma/                 # Persistent vector database files
│   └── cinemind.db             # Persistent SQLite database
│
└── logs/
    └── cinemind.log            # Execution and audit logs
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- **Python 3.11** installed.
- Recommended: [uv](https://docs.astral.sh/uv/) for blazing fast virtualenv and package management.

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd cinemind-ai

# Create Python 3.11 virtual environment
uv venv --python 3.11 .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to supply your Google Gemini API key:
```ini
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHROMA_PATH=storage/chroma
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=10
FINAL_TOP_K=4
```
*(Note: You can also enter or change your `GEMINI_API_KEY` directly in the Streamlit UI sidebar.)*

### 4. Ingest Primary Knowledge Source
Ensure `Quentin-Tarantino.pdf` is placed in `data/source/Quentin-Tarantino.pdf`, then run:
```bash
python scripts/ingest_book.py
```
This extracts all 277 pages, builds the chapter index, computes embeddings, and stores 268 vectors in persistent ChromaDB.

### 5. Launch CineMind AI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Empirical Evaluation & Benchmark Report

CineMind AI includes an automated evaluation framework (`evaluation/evaluate.py`) tested against 7 verified ground-truth questions grounded directly in _Quentin Tarantino_:

```
===========================================================================
📊 AGGREGATE EVALUATION REPORT
===========================================================================
  Total Benchmark Queries:   7
  Retrieval Precision:       60.7%
  Retrieval Recall:          54.3%
  Context Precision (MRR):   92.9%
  Context Recall:            39.4%
  Citation Correctness:      100.0%
  Answer Relevancy:          53.6%
  Average Response Latency:  760.6 ms
===========================================================================
```

### Key Findings
1. **Context Precision (MRR = 92.9%)**: The Cross-Encoder consistently placed the most authoritative passage containing the exact answer in the **#1 ranked spot**.
2. **Citation Correctness (100.0%)**: Every cited source page mapped strictly to genuine page numbers within the 277-page book. Zero hallucinated citations.
3. **Sub-second Hybrid Retrieval**: Full pipeline (Dense Vector Search + BM25 + RRF + Cross-Encoder reranking) completes in an average of **760 milliseconds**.

To re-run the benchmark suite:
```bash
python evaluation/evaluate.py
```

---

## 🧪 Automated Test Suite

Run all automated unit and integration tests:
```bash
pytest -v
```

Test coverage includes:
- `tests/test_pdf_loader.py`: 4 tests (PDF loading, text dehyphenation, chapter mapping, paragraph splitting)
- `tests/test_embeddings.py`: 2 tests (SentenceTransformer dimensions, ChromaDB persistent upsert/query/delete)
- `tests/test_retriever.py`: 3 tests (BM25 keyword search, Hybrid RRF fusion, Cross-Encoder reranking)
- `tests/test_rag.py`: 2 tests (End-to-end RAG query, citation generation, conversation SQLite persistence, duplicate prevention)

**Result**: 11 passed (100% pass rate).

---

## 🛡️ Responsible AI & Integrity Safeguards

- **Evidence Boundary**: If the retrieved documents do not contain the answer, CineMind AI states that the available sources do not provide sufficient information, avoiding hallucinations.
- **Untrusted Document Defense**: Text within uploaded documents is treated purely as passive data; embedded prompt injections or system overrides are safely neutralized.
- **No Hardcoded Keys**: API keys are managed securely through `.env` or password-masked session state.

---

## 📜 License
Developed as a production-grade Generative AI cinema intelligence project.
All book contents referenced belong to their respective copyright holders.
