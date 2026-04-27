from pathlib import Path

# Repo root (…/erp-ai-copilot)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
BLUEPRINT_CUSTOM_DOCS_DIR = DATA_DIR / "custom_docs/Blueprint"        # Custom PDFs (not committed)
USER_GUIDES_DIR = DATA_DIR / "custom_docs/UserGuides"
OFFICIAL_DOCS_DIR = DATA_DIR / "official_docs"    # Official PDFs (not committed)

VECTORSTORE_DIR = DATA_DIR / "vectorstores"       # FAISS indexes (not committed)
CUSTOM_INDEX_DIR = VECTORSTORE_DIR / "custom_mrp"     # rename per domain later
OFFICIAL_INDEX_DIR = VECTORSTORE_DIR / "official_mrp" # rename per domain later

BLUEPRINT_CHUNK_SIZE = 800
BLUEPRINT_CHUNK_OVERLAP = 150

USER_GUIDES_CHUNK_SIZE = 300
USER_GUIDES_CHUNK_OVERLAP = 100

#EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL = "intfloat/multilingual-e5-base"

DOMAIN_PAGE_RANGES = [
    # domain, start_page_inclusive, end_page_inclusive (1-based human pages)
    ("mrp", 7, 10),
    ("mps", 11, 15),
    ("wms", 16, 23),
    ("mrp", 24, 42),
    ("mps", 43, 62),
    ("wms", 63, 115),
]


def assert_dirs_exist():
    BLUEPRINT_CUSTOM_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    USER_GUIDES_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
