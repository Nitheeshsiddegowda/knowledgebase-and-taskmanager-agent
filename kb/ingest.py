import os
import uuid
from typing import List
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "vector_store"
COLLECTION_NAME = "kb_docs"


def ensure_collection():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    try:
        col = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        col = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
    return col




def _pdf_to_pages(file_bytes: bytes, filename: str):
    reader = PdfReader(file_bytes)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield {
            "source": filename,
            "page": i,
            "text": text,
        }




def _chunk(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c.strip()]




def ingest_uploaded_pdfs(collection, uploaded_files) -> int:
    docs, metas, ids = [], [], []
    count = 0
    for up in uploaded_files:
        filename = up.name
        for page in _pdf_to_pages(up, filename):
            for c in _chunk(page["text"], size=1000, overlap=200):
                docs.append(c)
                metas.append({"source": filename, "page": page["page"]})
                ids.append(str(uuid.uuid4()))
        count += 1
    if docs:
        collection.add(documents=docs, metadatas=metas, ids=ids)
    return count