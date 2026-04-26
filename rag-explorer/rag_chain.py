"""
RAG chain: retrieves relevant chunks from ChromaDB and generates answers using Groq.
"""
import os
import chromadb
import requests
from dotenv import load_dotenv
from embeddings import embed_query

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ecommerce_docs"
TOP_K = 5

# Groq API config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "gemma2-9b-it")


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-k relevant chunks for a query."""
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i],
        })
    return chunks


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate answer using Groq API with retrieved context."""
    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""You are a helpful e-commerce customer support assistant for ShopSmart.
Use the following context to answer the customer's question accurately.
If the answer is not in the context, say so honestly.

Context:
{context}

Customer Question: {query}

Answer:"""

    if not GROQ_API_KEY:
        return _fallback_generate(query, context_chunks)

    try:
        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful e-commerce assistant for ShopSmart."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[rag_chain] Groq API error: {e}")
        return _fallback_generate(query, context_chunks)


def _fallback_generate(query: str, context_chunks: list[dict]) -> str:
    """Simple extractive fallback when no LLM API is available."""
    if not context_chunks:
        return "I don't have enough information to answer that question."
    best = context_chunks[0]
    return f"Based on our documentation: {best['text'][:500]}"


def rag_query(query: str) -> dict:
    """Full RAG pipeline: retrieve + generate."""
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    return {
        "query": query,
        "answer": answer,
        "sources": chunks,
    }
