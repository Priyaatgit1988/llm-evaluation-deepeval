"""
Embedding module using Nomic Embed via sentence-transformers.
Falls back to a lighter model if Nomic is unavailable.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
FALLBACK_MODEL = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
            print(f"[embeddings] Loaded {MODEL_NAME}")
        except Exception as e:
            print(f"[embeddings] Failed to load {MODEL_NAME}: {e}")
            print(f"[embeddings] Falling back to {FALLBACK_MODEL}")
            _model = SentenceTransformer(FALLBACK_MODEL)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    # Nomic expects "search_document: " prefix for documents
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(prefixed, show_progress_bar=True)
    return embeddings.tolist()

def embed_query(query: str) -> list[float]:
    model = get_model()
    prefixed = f"search_query: {query}"
    embedding = model.encode([prefixed])
    return embedding[0].tolist()
