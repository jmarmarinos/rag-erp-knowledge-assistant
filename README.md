# rag-erp-knowledge-assistant
Enterprise RAG assistant that extracts operational knowledge from ERP documentation using embeddings, vector search, and LLMs.

AI-powered Retrieval-Augmented Generation (RAG) system designed to extract knowledge from complex ERP documentation and assist users in performing tasks in systems such as Microsoft Dynamics 365 Business Central especially in areas which have been customized and not covered by the official vendor's documentation.

The project focuses on transforming unstructured ERP documentation (PDFs, PowerPoints, manuals) into a searchable knowledge system that can answer operational questions using domain-specific context.

Example questions the system aims to answer:

How do I create a production order in Business Central?

How does MRP planning work in Business Central?

What steps are required to configure warehouse picking?

Architecture Overview

The system follows a modular RAG architecture designed to evolve from a local development environment to an enterprise cloud deployment.

Current architecture:

ERP Documentation
(PDF, PPT, manuals)
        ▼
Document Ingestion
(PyPDFLoader)
        ▼
Metadata Enrichment
(source, domain, doc_type, page)
        ▼
Text Chunking
(RecursiveCharacterTextSplitter)
        ▼
Embeddings
(sentence-transformers)
        ▼
Vector Database
FAISS
        ▼
Retriever
(similarity search + metadata filters)
        ▼
LLM Response Generation
(OpenAI / Azure OpenAI / local models)


Key Features:
Document Ingestion
  Loads ERP documentation and attaches structured metadata.
Supported sources:
  ERP blueprints
  user manuals
  Vendor's Documentation

The ingestion process automatically assigns domain metadata based on page ranges.
Documents are split into smaller chunks to improve retrieval accuracy.

Advantages:
  fast
  local execution
  no API cost
  good baseline for semantic search

Development Status
Current implementation includes:
  document ingestion
  metadata enrichment
  domain detection
  chunking pipeline
  vector database preparation
Next steps:
  embedding pipeline
  FAISS indexing
  retrieval layer
  LLM integration
  API interface

Technologies Used:
  Python
  LangChain
  FAISS
  Sentence Transformers
  PyPDF
  FastAPI (planned)
  Azure AI (planned)
