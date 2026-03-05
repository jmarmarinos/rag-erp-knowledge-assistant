from app.vectorstores import load_vectorstore
from app.config import CUSTOM_DOCS_DIR, OFFICIAL_DOCS_DIR


#load_vectorstore(CUSTOM_DOCS_DIR, "blueprint" )

custom_docs = load_vectorstore(CUSTOM_DOCS_DIR, "blueprint")

for d in custom_docs[::3]:
    print("\n" + "="*80)
    print("File:", d.metadata["file_name"])
    print("Page:", d.metadata["page_1based"])
    print("Domain:", d.metadata["domain"])
    print("-"*80)
    print(d.page_content)
