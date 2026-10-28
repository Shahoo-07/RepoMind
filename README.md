# Codebase RAG — Agentic Code Intelligence Platform

An AI-powered codebase intelligence system that enables natural language interaction with GitHub repositories. This project implements an advanced Agentic Retrieval-Augmented Generation (RAG) pipeline augmented with a knowledge graph to analyze, retrieve, and reason over code using semantic search, hybrid retrieval, and structural dependency analysis.

---

## Overview

Understanding large and unfamiliar codebases is complex due to fragmented logic, implicit dependencies, and limited documentation. This project addresses that challenge by building a Codebase RAG system that:

* Indexes entire repositories and parses code into structured units such as functions, classes, and modules
* Constructs a knowledge graph representing function call relationships and module dependencies
* Rewrites user queries to improve retrieval precision
* Performs hybrid retrieval combining semantic vector search and keyword-based BM25 search
* Applies re-ranking to refine relevance of retrieved context
* Utilizes an agent-based LLM workflow to perform multi-step reasoning over both retrieved code and graph structure
* Provides an interactive interface for exploring and querying codebases

---

## Key Features

### Intelligent Code Search

* Vector similarity search using dense embeddings
* Keyword-based retrieval using BM25
* Hybrid retrieval via Reciprocal Rank Fusion (RRF)
* Cross-encoder re-ranking for improved relevance

### Query Rewriting

* Rewrites user queries into more specific and retrieval-friendly forms
* Improves precision and reduces ambiguity in semantic search
* Enhances overall answer quality by retrieving more relevant context

### Knowledge Graph Integration

* Builds a graph of function-to-function call relationships and module-level dependencies
* Enables dependency analysis and impact analysis queries
* Supports multi-hop reasoning across code components
* Enhances retrieval by expanding context using graph neighbors

### Agentic RAG Workflow

* Multi-step reasoning using a LangGraph-based agent
* Tool-driven execution combining retrieval and graph queries
* Context-aware responses grounded in code with explicit source citations

### Code-Aware Ingestion

* Repository ingestion using GitPython
* AST parsing using Tree-sitter
* Function and class-level chunking for precise and structured retrieval

### End-to-End System

* FastAPI backend for orchestration and query handling
* Streamlit frontend for interactive exploration
* Docker-based deployment for portability and scalability

---

## System Architecture

```text
GitHub Repository
    ↓
Code Ingestion (GitPython)
    ↓
AST Parsing (Tree-sitter)
    ↓
Chunking (functions, classes, modules)
    ↓
Embedding Generation
    ↓
LanceDB Vector Store
    ↓
Knowledge Graph Construction (NetworkX)
    ↓
Query Rewriting
    ↓
Hybrid Retrieval (Vector + BM25)
    ↓
Re-ranking (Cross Encoder)
    ↓
Agentic Reasoning (LangGraph)
    ↓
LLM
    ↓
Final Answer with Source Citations
```

---

## Tech Stack

| Component       | Technology                             |
| --------------- | -------------------------------------- |
| LLM             | Ollama (Llama, Mistral)                |
| Embeddings      | Sentence Transformers / EmbeddingGemma |
| Vector Database | LanceDB                                |
| Parsing         | Tree-sitter                            |
| Knowledge Graph | NetworkX + Python AST                  |
| Retrieval       | BM25 + Vector Search + RRF             |
| Re-ranking      | Cross-encoder (sentence-transformers)  |
| Agent Framework | LangGraph (ReAct-style agent)          |
| Backend         | FastAPI                                |
| Frontend        | Streamlit                              |
| Deployment      | Docker                                 |

---

## Example Use Cases

* Explain the overall structure and architecture of a repository
* Analyze function dependencies and module relationships
* Perform impact analysis for code changes
* Locate specific functions, classes, or API endpoints
* Understand authentication or data flow across modules
* Generate documentation directly from source code
* Explore and navigate unfamiliar codebases efficiently
