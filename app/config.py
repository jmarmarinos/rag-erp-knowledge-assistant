from pathlib import Path
# --------- CONFIG ---------
PROJECT_ROOT = Path(__file__).resolve().parent  # folder where this script lives
 
CUSTOM_DOCS_DIR = PROJECT_ROOT / ".." / "data" / "custom_docs"      # Custom PDFs 
OFFICIAL_DOCS_DIR = PROJECT_ROOT / "data" / "official_docs"  # Official PDFs here

VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstores"
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
    ("wms", 63, 115)
]
# --------------------------
