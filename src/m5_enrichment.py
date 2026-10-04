from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
from concurrent.futures import ThreadPoolExecutor
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, LLM_EXTRA_PARAMS, LLM_MAX_WORKERS, get_llm_client


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── LLM helpers (OpenAI hoặc Gemini — xem config.LLM_PROVIDER) ──

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = get_llm_client()
    return _client


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi LLM; trả "" nếu không có API key hoặc lỗi (để caller dùng fallback)."""
    if not LLM_API_KEY:
        return ""
    try:
        kwargs = {**LLM_EXTRA_PARAMS}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _get_client().chat.completions.create(
            model=LLM_MODEL, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI call failed: {e}")
        return ""


def _chat_json(system: str, user: str, max_tokens: int) -> dict:
    """Như _chat nhưng ép JSON mode (tránh lỗi model bọc JSON trong ```json ... ```)."""
    raw = _chat(system, user, max_tokens, json_mode=True)
    if not raw:
        return {}
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Invalid JSON from LLM: {e}")
        return {}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    result = _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.", text, max_tokens=150)
    if result:
        return result
    # Extractive fallback (không cần API): lấy 2 câu đầu
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    result = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
        text, max_tokens=200)
    if result:
        questions = [q.strip().lstrip("0123456789.-) ").strip() for q in result.split("\n")]
        return [q for q in questions if q][:n_questions]
    # Extractive fallback: biến câu thành câu hỏi
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
    if context:
        return f"{context}\n\n{text}"
    # Simple fallback: luôn giữ nguyên văn chunk gốc
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    result = _chat_json(
        'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
        '"category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150)
    if result:
        return result
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    return _chat_json(
        """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu (nêu tên chính sách/phiên bản nếu có)",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", max_tokens=400)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    def enrich_one(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
            if not isinstance(auto_meta, dict):
                auto_meta = {}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    # Gọi API song song (I/O-bound) — executor.map giữ nguyên thứ tự chunks
    enriched = []
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS if LLM_API_KEY else 1) as pool:
        for i, item in enumerate(pool.map(enrich_one, chunks)):
            enriched.append(item)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
