from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
import os
import certifi
import json

import numpy as np

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx2txt


Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

STORE_PATH = Path("data/rag_store.json")
_STORE_LOCK = Lock()


def _empty_store() -> dict:
    return {"threads": {}}


def _load_store() -> dict:
    if not STORE_PATH.exists():
        return _empty_store()

    try:
        with STORE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _empty_store()

    threads = data.get("threads", {})
    if not isinstance(threads, dict):
        threads = {}

    return {"threads": threads}


def _save_store(store: dict) -> None:
    temp_path = STORE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False)

    temp_path.replace(STORE_PATH)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)

    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if not denominator:
        return 0.0

    return float(np.dot(left_vector, right_vector) / denominator)



def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")




def add_document_to_rag(file_path: str, thread_id: str):
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)
    chunk_embeddings = embeddings.embed_documents(chunks)

    with _STORE_LOCK:
        store = _load_store()
        thread_docs = store["threads"].setdefault(thread_id, [])

        for chunk, embedding in zip(chunks, chunk_embeddings):
            thread_docs.append(
                {
                    "source": Path(file_path).name,
                    "page_content": chunk,
                    "embedding": embedding
                }
            )

        _save_store(store)

    return {
        "filename": Path(file_path).name,
        "chunks": len(chunks)
    }





def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    query_embedding = embeddings.embed_query(query)

    with _STORE_LOCK:
        store = _load_store()
        thread_docs = store["threads"].get(thread_id, [])

    scored_docs = []

    for item in thread_docs:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            continue

        scored_docs.append(
            (
                _cosine_similarity(query_embedding, embedding),
                item
            )
        )

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    docs = [item[1] for item in scored_docs[:k]]

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.get('page_content', '')}"
        )

    return "\n\n".join(results)