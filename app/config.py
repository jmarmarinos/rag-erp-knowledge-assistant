from pathlib import Path

# Repo root (…/erp-ai-copilot)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CUSTOM_DOCS_DIR = DATA_DIR / "custom_docs"        # Custom PDFs (not committed)
OFFICIAL_DOCS_DIR = DATA_DIR / "official_docs"    # Official PDFs (not committed)

VECTORSTORE_DIR = DATA_DIR / "vectorstores"       # FAISS indexes (not committed)
CUSTOM_INDEX_DIR = VECTORSTORE_DIR / "custom_mrp"     # rename per domain later
OFFICIAL_INDEX_DIR = VECTORSTORE_DIR / "official_mrp" # rename per domain later

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOMAIN_PAGE_RANGES = [
    # domain, start_page_inclusive, end_page_inclusive (1-based human pages)
    ("mrp", 7, 10),
    ("mps", 11, 15),
    ("wms", 16, 23),
    ("mrp", 24, 42),
    ("mps", 43, 62),
    ("wms", 63, 115),
]
