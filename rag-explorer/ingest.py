"""
Document ingestion pipeline: reads PDF/TXT files, chunks them, and stores in ChromaDB.
"""
import os
import re
import chromadb
from chromadb.config import Settings
from embeddings import embed_texts

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ecommerce_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DATA_DIR = "./data"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count, respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap // 5:]) if len(words) > overlap // 5 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(filepath: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except ImportError:
        print("[ingest] PyPDF2 not installed, skipping PDF")
        return ""


def ingest_documents():
    """Read all documents from DATA_DIR, chunk, embed, and store in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists, then recreate
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_ids = []
    all_metadata = []

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            text = read_txt(filepath)
        elif filename.endswith(".pdf"):
            text = read_pdf(filepath)
        else:
            continue

        print(f"[ingest] Processing {filename} ({len(text)} chars)")
        chunks = chunk_text(text)
        print(f"[ingest] Generated {len(chunks)} chunks from {filename}")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({"source": filename, "chunk_index": i})

    if not all_chunks:
        print("[ingest] No documents found to ingest!")
        return []

    print(f"[ingest] Embedding {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks)

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    print(f"[ingest] Stored {len(all_chunks)} chunks in ChromaDB")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_documents()
    print(f"\nIngestion complete. Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---\n{c[:200]}...")
