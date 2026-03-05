from fastapi import FastAPI

app = FastAPI(title="ERP AI Copilot (WIP)")

@app.get("/health")
def health():
    return {"status": "ok", "note": "RAG pipeline WIP; ingestion + chunking implemented."}
