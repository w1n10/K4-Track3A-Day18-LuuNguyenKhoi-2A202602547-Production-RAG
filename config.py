"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- LLM provider ---
# Gemini dùng OpenAI-compatible endpoint → vẫn dùng SDK `openai` / `langchain_openai`.
# Mặc định: có GEMINI_API_KEY → gemini, ngược lại → openai.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "openai").lower()
if LLM_PROVIDER == "gemini":
    LLM_API_KEY = GEMINI_API_KEY
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "gemini-embedding-001")
else:
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = None
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")
# Gemini 2.5 mặc định bật "thinking" (thinking tokens ăn vào max_tokens) → tắt bằng "none".
# Gemini 3.x Flash-Lite mặc định thinking_level=minimal và không nhận "none" → không gửi tham số.
LLM_EXTRA_PARAMS = ({"reasoning_effort": "none"}
                    if LLM_PROVIDER == "gemini" and LLM_MODEL.startswith("gemini-2.5") else {})
# Free tier Gemini giới hạn request/phút → giảm số luồng gọi song song khi enrichment
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "2" if LLM_PROVIDER == "gemini" else "8"))


def get_llm_client():
    """OpenAI SDK client trỏ tới provider đang chọn (OpenAI hoặc Gemini)."""
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=5, timeout=60)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
