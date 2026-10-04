from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, LLM_API_KEY, LLM_MODEL, LLM_EXTRA_PARAMS, get_llm_client

ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý chính sách nội bộ công ty. Trả lời CHỈ dựa trên context được cung cấp.
Quy tắc:
- Nếu context có nhiều phiên bản của cùng một chính sách, dùng phiên bản MỚI NHẤT (hiện hành) và nói rõ phiên bản cũ đã bị thay thế.
- Với câu hỏi Có/Không, mở đầu câu trả lời bằng "Có" hoặc "Không".
- Với câu hỏi cần tính toán hoặc kết hợp nhiều quy định, nêu ngắn gọn các con số và phép tính lấy từ context.
- Trả lời ngắn gọn, đi thẳng vào câu hỏi, bằng tiếng Việt. Không thêm thông tin ngoài context.
- Nếu context không có thông tin → trả lời đúng một câu: "Không tìm thấy."."""

# Thời gian build (giây) và từng query (ms) — dùng cho latency breakdown report
BUILD_LATENCY: dict[str, float] = {}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_texts: dict[str, str] = {}
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_texts[parent.metadata["parent_id"]] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    BUILD_LATENCY["chunking"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} child chunks / {len(parent_texts)} parents from {len(docs)} documents "
          f"({BUILD_LATENCY['chunking']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    BUILD_LATENCY["enrichment"] = time.time() - t0

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_texts = parent_texts  # child → parent lookup cho run_query()
    BUILD_LATENCY["indexing"] = time.time() - t0
    print(f"  ✓ Indexed ({BUILD_LATENCY['indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load ngay để latency query không tính thời gian load model
    BUILD_LATENCY["reranker_load"] = time.time() - t0
    print(f"  ✓ Reranker ready ({BUILD_LATENCY['reranker_load']:.1f}s)", flush=True)

    return search, reranker


def _expand_to_parents(reranked, parent_texts: dict[str, str], top_k: int) -> list[str]:
    """Retrieve child (precision) → return parent (context). Bỏ trùng parent, giữ thứ tự rerank."""
    contexts, seen = [], set()
    for r in reranked:
        pid = r.metadata.get("parent_id")
        key = pid if pid in parent_texts else r.text
        if key in seen:
            continue
        seen.add(key)
        contexts.append(parent_texts.get(pid, r.text))
        if len(contexts) == top_k:
            break
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              timings: dict | None = None) -> tuple[str, list[str]]:
    """Run single query through pipeline. Nếu truyền `timings`, ghi latency (ms) từng bước vào đó."""
    timings = timings if timings is not None else {}

    t0 = time.perf_counter()
    results = search.search(query)
    timings["search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rerank toàn bộ candidates, sau đó lấy top-k parent KHÁC NHAU (nhiều child có thể chung parent)
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    parent_texts = getattr(search, "parent_texts", {})
    if reranked:
        contexts = _expand_to_parents(reranked, parent_texts, RERANK_TOP_K)
    else:
        contexts = [r.text for r in results[:RERANK_TOP_K]]

    t0 = time.perf_counter()
    if LLM_API_KEY and contexts:
        try:
            context_str = "\n\n---\n\n".join(contexts)
            resp = get_llm_client().chat.completions.create(model=LLM_MODEL, temperature=0, messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ], **LLM_EXTRA_PARAMS)
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["llm_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = timings["search_ms"] + timings["rerank_ms"] + timings["llm_ms"]
    return answer, contexts


def report_latency(query_timings: list[dict], path: str = "reports/latency_report.json") -> dict:
    """In bảng latency breakdown từng bước + lưu JSON."""
    def _stats(values: list[float]) -> dict:
        s = sorted(values)
        return {"avg": round(sum(s) / len(s), 1), "p50": round(s[len(s) // 2], 1),
                "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1), "max": round(s[-1], 1)}

    per_step = {step: _stats([t[step] for t in query_timings])
                for step in ["search_ms", "rerank_ms", "llm_ms", "total_ms"]} if query_timings else {}
    report = {"build_seconds": {k: round(v, 2) for k, v in BUILD_LATENCY.items()},
              "query_ms": per_step, "num_queries": len(query_timings)}

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print("Build (one-time):")
    for step, sec in report["build_seconds"].items():
        print(f"  {step:<15} {sec:>8.2f}s")
    print(f"Per query ({len(query_timings)} queries):")
    print(f"  {'Step':<12} {'avg':>9} {'p50':>9} {'p95':>9} {'max':>9}")
    for step, s in per_step.items():
        print(f"  {step.replace('_ms', ''):<12} {s['avg']:>7.1f}ms {s['p50']:>7.1f}ms {s['p95']:>7.1f}ms {s['max']:>7.1f}ms")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_timings = []

    for i, item in enumerate(test_set):
        timings = {}
        answer, contexts = run_query(item["question"], search, reranker, timings)
        query_timings.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}... ({timings['total_ms']:.0f}ms)", flush=True)

    report_latency(query_timings)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
