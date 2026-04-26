"""
Configuration for DeepEval framework.
Supports switching between local (Ollama) and cloud LLMs for both judge and generator roles.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── LLM Selection ───
JUDGE_LLM = os.getenv("JUDGE_LLM", "groq")
GENERATOR_LLM = os.getenv("GENERATOR_LLM", "groq")

# ─── Model Names ───
MODELS = {
    "openai": {"name": "gpt-4o", "type": "cloud", "base_url": "https://api.openai.com/v1"},
    "grok": {"name": "grok-3-mini", "type": "cloud", "base_url": "https://api.x.ai/v1"},
    "groq": {"name": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "type": "cloud", "base_url": "https://api.groq.com/openai/v1"},
    "groq_oss120b": {"name": "openai/gpt-oss-120b", "type": "cloud", "base_url": "https://api.groq.com/openai/v1"},
    "groq_scout": {"name": "meta-llama/llama-4-scout-17b-16e-instruct", "type": "cloud", "base_url": "https://api.groq.com/openai/v1"},
    "groq_qwen": {"name": "qwen/qwen3-32b", "type": "cloud", "base_url": "https://api.groq.com/openai/v1"},
    "oss_120b": {"name": "mistral-large-latest", "type": "local"},
    "gemma": {"name": "gemma3:1b", "type": "local"},
}

# ─── RAG Explorer ───
RAG_EXPLORER_URL = os.getenv("RAG_EXPLORER_URL", "http://localhost:5001")
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:3000")
