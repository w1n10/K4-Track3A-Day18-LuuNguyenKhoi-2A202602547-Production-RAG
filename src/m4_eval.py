from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _safe_float(value) -> float:
    """float() an toàn: None/NaN → 0.0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {m: 0.0 for m in METRICS}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from config import (LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                            LLM_EXTRA_PARAMS, EVAL_EMBEDDING_MODEL)

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": [list(c) for c in contexts], "ground_truth": ground_truths,
        })
        # Chỉ định judge LLM + embeddings theo provider (default của RAGAS 0.1 là OpenAI ada-002)
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=LLM_API_KEY,
                         base_url=LLM_BASE_URL, max_retries=6, model_kwargs=LLM_EXTRA_PARAMS)
        # check_embedding_ctx_length=False: gửi text thay vì token ids (Gemini endpoint chỉ nhận text)
        embeddings = OpenAIEmbeddings(model=EVAL_EMBEDDING_MODEL, api_key=LLM_API_KEY,
                                      base_url=LLM_BASE_URL, check_embedding_ctx_length=False)
        from ragas.run_config import RunConfig

        run_config = RunConfig()
        if LLM_PROVIDER == "gemini":
            # answer_relevancy mặc định xin n=3 completions trong 1 request — Gemini endpoint có thể không hỗ trợ n>1
            answer_relevancy.strictness = 1
            # Free tier giới hạn request/phút → ít luồng song song, retry lâu hơn khi bị 429
            run_config = RunConfig(max_workers=4, max_retries=10, max_wait=90, timeout=180)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=llm, embeddings=embeddings, run_config=run_config)
        df = result.to_pandas()

        per_question = []
        for i, row in df.iterrows():
            per_question.append(EvalResult(
                question=questions[i], answer=answers[i],
                contexts=list(contexts[i]), ground_truth=ground_truths[i],
                **{m: _safe_float(row.get(m)) for m in METRICS},
            ))
        # RAGAS có thể trả NaN cho vài câu (LLM parse lỗi) → tính như 0 để JSON hợp lệ
        aggregate = {m: _safe_float(df[m].fillna(0).mean()) if m in df else 0.0 for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}



def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "scores": {m: round(s, 4) for m, s in scores.items()},
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
