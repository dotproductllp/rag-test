# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

Required environment variables:
```
OPENAI_API_KEY       # OpenAI API key
PINECONE_API_KEY     # Pinecone API key
INDEX_NAME           # Pinecone index name
COSMOS_URL           # Azure Cosmos DB endpoint
COSMOS_KEY           # Azure Cosmos DB key
```

## Running the Scripts

**PDF Q&A:**
```bash
python pdf_ingest_to_pinecone.py   # Ingest PDF into Pinecone
python semantic_search_ranking.py  # Semantic search over PDFs
python chat.py                     # Interactive RAG chatbot for PDFs
```

**Movie database (CSV):**
```bash
python film_search/csv_ingest_to_pinecone.py
python film_search/search_rank.py
```

**LinkedIn posts (JSON):**
```bash
python linkedin-posts/ingest_json.py   # Ingest posts into Pinecone
python linkedin-posts/search_rank.py
python linkedin-posts/post_chat.py
```

**LinkedIn posts → Azure Cosmos DB pipeline:**
```bash
python post_processing/clean_data.py      # Clean/normalize raw post text
python post_processing/store_data.py      # Upload with embeddings to Cosmos DB
python post_processing/hybrid_search.py  # Run hybrid vector + keyword search
python post_processing/llm_search_query.py  # LLM-powered query analysis
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system with two storage backends: **Pinecone** (pure vector search) and **Azure Cosmos DB** (hybrid vector + keyword search).

### Data flow

```
Raw data (PDF / CSV / JSON)
  → Text cleaning (post_processing/clean_data.py)
  → Embedding via text-embedding-3-small
  → Storage: Pinecone  OR  Azure Cosmos DB
  → Query time: hybrid_search.py / llm_search_query.py
  → Answer generation via LangGraph (chat.py / post_chat.py)
```

### LangGraph chatbot pattern

Both `chat.py` and `linkedin-posts/post_chat.py` implement the same two-node graph:

```
START → retrieve (vector similarity search) → generate (GPT response) → END
```

### Azure Cosmos DB specifics

- Async/await with concurrency limit of 32 (`post_processing/store_data.py`)
- DiskANN vector index; partition key is `date_published`
- Batch size: 10,000 documents per bulk upload
- `post_processing/hybrid_search.py` combines vector similarity with keyword filters

### LLM query analyzer (`post_processing/llm_search_query.py`)

Decomposes a natural-language user prompt into:
- `search_query` — free-text semantic query passed to the vector index
- `keywords` — strict filter terms applied as keyword constraints in hybrid search

### Metadata constraints

- **Pinecone**: metadata must be flat (strings, numbers, booleans, or lists of strings). Nested dicts are serialized to JSON strings before upsert.
- **Cosmos DB**: nested documents are fine; use `date_published` as the partition key.

### Text cleaning (`post_processing/clean_data.py`)

Uses `multiprocessing` (CPU-bound). Key normalizations: fancy Unicode → ASCII, URLs/emails/UUIDs → placeholders, hashtag removal, non-Latin post filtering, whitespace normalization.

## Models used

| Purpose | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Chat / generation | `gpt-4o` (or as configured per script) |

> Check each script's model constant if you need to swap models — they are defined at the top of each file, not in a central config.
