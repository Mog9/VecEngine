# VecEngine

VecEngine is a lightweight, from-scratch vector retrieval system that implements **semantic similarity search**, **persistent vector storage**, and **coarse indexing** without relying on external vector databases or cloud services.

The project is designed for **clarity, correctness, and experimentation**, making every stage of the retrieval pipeline explicit and inspectable.

---

## What VecEngine Is

VecEngine provides a minimal but complete retrieval stack:

- Dense embedding–based semantic search
- Exact and vectorized batch similarity computation
- Disk-backed vector storage with deduplication
- Optional coarse indexing for reduced query-time computation
- Clean separation between storage, retrieval, and indexing logic

It is intended for **local or small-scale retrieval workloads**, educational use, and rapid prototyping, not production-scale deployment.

---

## Core Components

### 1. Retrieval Engine

The retrieval engine performs semantic similarity search using dense embeddings and cosine similarity.

**Capabilities:**
- Query encoding using `SentenceTransformer`
- Exact full-scan retrieval with cosine similarity
- Top-k ranking of results
- Threshold-based filtering
- Query-level LRU caching to avoid repeated computation

**Batch Retrieval:**
- Supports vectorized batch search using matrix multiplication
- Computes all query–document similarities in a single NumPy operation
- Suitable for evaluating multiple queries efficiently

---

### 2. Vector Store (Persistence Layer)

VecEngine includes a lightweight, disk-backed vector store responsible for managing embedding records.

**Features:**
- Persistent storage via JSON
- Deterministic ID generation for vectors
- Duplicate-aware ingestion across multiple data sources
- Full CRUD support (add, update, delete)
- Explicit separation from retrieval and indexing logic

This layer focuses on **data integrity and lifecycle management**, not retrieval performance.

---

### 3. Indexing (Coarse Routing)

VecEngine explores cluster-based indexing to reduce query-time computation.

**Indexing Strategy:**
- Offline KMeans clustering over stored vectors
- Query-time routing to the nearest cluster centroid
- Exact cosine similarity computed only within the selected cluster

This implements a **coarse-to-fine retrieval pattern**, trading recall for reduced compute.

> Note: Indexing is optional and experimental; exact full-scan retrieval remains available.

---

## Retrieval Modes

VecEngine supports multiple retrieval paths:

- **Exact Retrieval:**  
  Full cosine similarity scan over all vectors (O(N))

- **Batch Retrieval:**  
  Vectorized similarity computation using matrix multiplication

- **Clustered Retrieval:**  
  Coarse routing via KMeans followed by exact scoring within a subset

Each mode is implemented explicitly to make tradeoffs transparent.

---

## Design Principles

- **No hidden abstractions** — every operation is explicit
- **Separation of concerns** — storage, retrieval, and indexing are independent
- **Honest constraints** — CPU-only, small-scale, local workloads
- **Inspectability** — code favors readability over premature optimization

---

## Use Cases

- Learning how vector search systems work internally
- Prototyping retrieval pipelines without external dependencies
- Small-scale semantic search experiments
- Educational exploration of ANN tradeoffs

---

## Summary

VecEngine demonstrates that a functional vector retrieval system can be built from first principles using:

- Dense embeddings
- Linear algebra
- Explicit data management
- Simple but effective indexing strategies

The project emphasizes **correctness, transparency, and systems-level understanding** over production complexity.
