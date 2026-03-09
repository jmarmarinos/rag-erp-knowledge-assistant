from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import (
    CUSTOM_DOCS_DIR,
    OFFICIAL_DOCS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOMAIN_PAGE_RANGES,
    assert_dirs_exist  
)


def detect_source(folder: Path) -> str:
    """Derive source label based on the folder path."""
    if folder.resolve() == CUSTOM_DOCS_DIR.resolve():
        return "custom"
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
            d.metadata["domain"] = detect_domain(page1)

        docs.extend(pages)

    print(f"Loaded {len(docs)} pages from {len(pdfs)} PDFs in: {folder}")
    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
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
        print(d.page_content[:300])  


if __name__ == "__main__":
    assert_dirs_exist()

    # --- Step 1: Load ---
    custom_pages = load_pdfs_with_metadata(CUSTOM_DOCS_DIR, doc_type="blueprint")
    official_pages = load_pdfs_with_metadata(OFFICIAL_DOCS_DIR, doc_type="official")

    # Optional: inspect a few pages
    inspect_docs(custom_pages, n=2)

    # --- Step 2: Split ---
    custom_chunks = split_docs(custom_pages)
    official_chunks = split_docs(official_pages)

    # Optional: inspect a few chunks
    inspect_docs(custom_chunks, n=2)

    print("Loader + splitter steps completed.")
