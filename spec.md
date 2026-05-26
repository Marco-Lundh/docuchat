# docuchat — Project Specification

## Purpose

docuchat is a CLI tool for chatting with PDF documents via RAG
(Retrieval-Augmented Generation). The user loads one or more PDFs that are
indexed locally, then asks questions in natural language and receives answers
generated from the document content.

## Tech Stack

| Component       | Technology                                                  |
|-----------------|-------------------------------------------------------------|
| LLM             | Groq API — `llama-3.3-70b-versatile`                        |
| Embeddings      | sentence-transformers `all-MiniLM-L6-v2` (local, free)     |
| Vector database | FAISS (local, `IndexFlatIP` with L2 normalization)          |
| PDF parsing     | PyMuPDF (`fitz`)                                            |
| CLI / UI        | Rich (colored terminal output)                              |
| Package manager | uv                                                          |
| Python          | 3.14+                                                       |

## Architecture

```
docuchat/
├── src/
│   ├── main.py            # CLI entry point, argument parsing, chat loop
│   ├── ingest.py          # Reads PDFs, splits into chunks, builds FAISS index
│   ├── retriever.py       # Searches relevant chunks via embedding similarity
│   └── chat.py            # Sends question + context to Groq and returns answer
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── create_test_pdf.py  # Generates test PDF (Granit Software AB)
│   ├── test_chat.py
│   ├── test_ingest.py
│   ├── test_main.py
│   └── test_retriever.py
├── pyproject.toml     # UV project file with dependencies
└── .env               # API key (not in git)
```

Persisted files (created during indexing, not in git):
- `docuchat.index` — FAISS index
- `docuchat.chunks` — pickle list of text chunks

## Features

### Indexing
- Reads text from one or more PDF files using PyMuPDF
- Splits text into overlapping word-based chunks (500 words, 50-word overlap)
- Creates embeddings with `all-MiniLM-L6-v2` and stores them in FAISS

### Retrieval
- Encodes the user's question into an embedding
- Finds the 4 most relevant chunks using cosine similarity (L2-normalized dot product)
- Returns chunks as context to the LLM

### Generation
- Sends the question + retrieved chunks to the Groq API
- The model is instructed to answer based solely on the provided content
- Temperature 0.2 for consistent, fact-grounded responses

### Multi-document
- Multiple PDF files can be passed as arguments
- All are indexed together into a single shared FAISS index
- Retrieval searches across all documents simultaneously

### Index reset
- `--reset` flag deletes the existing index
- `--reset <file.pdf>` resets and immediately rebuilds with new documents

## CLI Usage

```bash
# Index and chat with a single PDF
uv run main.py document.pdf

# Chat with multiple PDFs
uv run main.py file1.pdf file2.pdf file3.pdf

# Reset the index
uv run main.py --reset

# Reset and rebuild the index
uv run main.py --reset document.pdf
```

## Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at: https://console.groq.com

## Design Decisions

- **Groq instead of Anthropic/OpenAI** — free tier with generous limits, no credit card required
- **Local sentence-transformers** — no API costs for embeddings, works offline after first download
- **Local FAISS** — simple setup, no external services, sufficient for documents up to thousands of pages
- **truststore** — required on Windows so Python trusts the system certificate store (corporate networks)
- **Word-based chunking** — simple and predictable, works well for prose-heavy documents
