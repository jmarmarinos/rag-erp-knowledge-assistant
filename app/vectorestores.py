from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    BLUEPRINT_CUSTOM_DOCS_DIR,
    USER_GUIDES_DIR,
    OFFICIAL_DOCS_DIR,
    BLUEPRINT_CHUNK_SIZE,
    BLUEPRINT_CHUNK_OVERLAP,
    USER_GUIDES_CHUNK_SIZE,
    USER_GUIDES_CHUNK_OVERLAP,
    DOMAIN_PAGE_RANGES,
    EMBED_MODEL,
    CUSTOM_INDEX_DIR,
    OFFICIAL_INDEX_DIR,
    assert_dirs_exist   
)


def detect_source(folder: Path) -> str:
    """Derive source label based on the folder path."""
    if folder.resolve() == BLUEPRINT_CUSTOM_DOCS_DIR.resolve():
        return "Blueprint"
    if folder.resolve() == USER_GUIDES_DIR.resolve():
        return "userguides"
    if folder.resolve() == OFFICIAL_DOCS_DIR.resolve():
        return "official"
    return "unknown"


def detect_domain(page_1based: int) -> str:
    """Map page number to a domain using DOMAIN_PAGE_RANGES."""
    for domain, start, end in DOMAIN_PAGE_RANGES:
        if start <= page_1based <= end:
            return domain
    return "unknown"


def load_pdfs_with_metadata(folder: Path, doc_type: str) -> List[Document]:
    """Load all PDFs from folder and attach metadata per page."""
    pdfs = sorted(folder.glob("*.pdf"))
    source = detect_source(folder)

    docs: List[Document] = []
    for pdf_path in pdfs:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # list[Document], one per page

        for d in pages:
            page0 = d.metadata.get("page", 0)   # 0-based
            page1 = page0 + 1                  # 1-based for humans

            d.metadata["source"] = source
            d.metadata["doc_type"] = doc_type
            d.metadata["file_name"] = pdf_path.name
            d.metadata["page_1based"] = page1
            if doc_type == 'blueprint':
                d.metadata["domain"] = detect_domain(page1)
            elif doc_type == 'userguide': 
                d.metadata["domain"] = 'wms'
        docs.extend(pages)

    print(f"Loaded {len(docs)} pages from {len(pdfs)} PDFs in: {folder}")
    return docs


def split_docs(CHUNK_SIZE, CHUNK_OVERLAP, docs: List[Document]) -> List[Document]:
    """Split documents into chunks for better retrieval."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    for i, c in enumerate(chunks[:5], start=1):
        print("\n" + "=" * 80)
        print("Chunk:", i)
        print("Metadata:", c.metadata)
        print("-" * 80)
        print(c.page_content[:800])
    
    return chunks


def inspect_docs(docs: List[Document], n: int = 3) -> None:
    """Print a few docs/chunks for manual inspection."""
    for d in docs[:n]:
        print("\n" + "=" * 80)
        print("File:", d.metadata.get("file_name"))
        print("Page:", d.metadata.get("page_1based"))
        print("Domain:", d.metadata.get("domain"))
        print("Source:", d.metadata.get("source"))
        print("Type:", d.metadata.get("doc_type"))
        print("-" * 80)
        print(d.page_content[:1200])  

def build_faiss_index(chunks: List[Document], out_dir: Path) -> FAISS:
    """Create a FAISS index from chunks and save it locally."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Chunks to index:", len(chunks))
    print("Faiss directory:", out_dir)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(out_dir))

    print(f"Saved FAISS index to: {out_dir}")
    return vectorstore

def test_similarity_search(
    vectorstore: FAISS,
    query: str,
    k: int = 3,
    doc_type: str | None = None,
    domain: str | None = None
):
    """Flexible similarity search with optional filters"""

    search_kwargs = {"k": k}

    filter_dict = {}

    if doc_type is not None:
        filter_dict["doc_type"] = doc_type

    if domain is not None:
        filter_dict["domain"] = domain

    if filter_dict:
        search_kwargs["filter"] = filter_dict

    results = vectorstore.similarity_search(query, **search_kwargs)

    print("Query:", query)
    print("Filters:", filter_dict)
    print("Results returned:", len(results))

    for i, doc in enumerate(results, start=1):
        print("\n" + "=" * 80)
        print("Result:", i)
        print("File:", doc.metadata.get("file_name"))
        print("Page:", doc.metadata.get("page_1based"))
        print("Metadata:", doc.metadata)
        print("-" * 80)
        print(doc.page_content[:800])

if __name__ == "__main__":
    assert_dirs_exist()

    # --- Step 1: Load ---
    custom_pages = load_pdfs_with_metadata(BLUEPRINT_CUSTOM_DOCS_DIR, doc_type="blueprint")
    user_guides = load_pdfs_with_metadata(USER_GUIDES_DIR, doc_type="userguide")
    official_pages = load_pdfs_with_metadata(OFFICIAL_DOCS_DIR, doc_type="official")

    # Optional: inspect a few pages
    inspect_docs(custom_pages, n=2)

    # --- Step 2: Split ---
    custom_chunks = split_docs(custom_pages)
    userguides_chunks = split_docs(user_guides)
    official_chunks = split_docs(official_pages)

    # Optional: inspect a few chunks
    inspect_docs(custom_chunks, n=2)

     # --- Step 3: Build FAISS ---
    custom_store = build_faiss_index(custom_chunks, CUSTOM_INDEX_DIR)
    userguides_store = build_faiss_index(userguides_chunks, CUSTOM_INDEX_DIR)
    official_store = build_faiss_index(official_chunks, OFFICIAL_INDEX_DIR)

    # --- Step 4: Retrieval test ---
    test_similarity_search(custom_store, "How does MRP planning work?", k=3)
    test_similarity_search(custom_store, "How do I process warehouse picking?", k=3)

    print("Loader + splitter steps completed.")
