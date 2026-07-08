import json
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from ask import load_vector_store, retrieve_context, generate_extractive_answer


EVAL_FILE = PROJECT_ROOT / "evaluation" / "eval_questions.json"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "eval_results.json"


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def evaluate_answer(answer: str, expected_keywords: list[str]) -> dict:
    answer_norm = normalize_text(answer)

    keyword_results = []

    for keyword in expected_keywords:
        keyword_norm = normalize_text(keyword)
        found = keyword_norm in answer_norm

        keyword_results.append({
            "keyword": keyword,
            "found": found
        })

    found_count = sum(item["found"] for item in keyword_results)
    total_count = len(expected_keywords)

    score = found_count / total_count if total_count > 0 else 0.0

    return {
        "score": score,
        "found_keywords": found_count,
        "total_keywords": total_count,
        "keyword_results": keyword_results,
        "passed": score >= 0.75
    }


def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE}")

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)

    print("=" * 80)
    print("CLINICAL RAG EVALUATION")
    print("=" * 80)

    print("\nLoading vector store...")
    db = load_vector_store()

    results = []

    for item in eval_questions:
        question_id = item["id"]
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        print("\n" + "-" * 80)
        print(f"Question ID: {question_id}")
        print(f"Question: {question}")

        retrieved_context = retrieve_context(db, question, final_k=5)
        answer = generate_extractive_answer(question, retrieved_context)

        eval_result = evaluate_answer(answer, expected_keywords)

        print("\nAnswer:")
        print(answer)

        print("\nExpected keywords:")
        for keyword_result in eval_result["keyword_results"]:
            status = "FOUND" if keyword_result["found"] else "MISSING"
            print(f"- {keyword_result['keyword']}: {status}")

        print(f"\nScore: {eval_result['score']:.2f}")
        print(f"Passed: {eval_result['passed']}")

        sources = []

        for context_item in retrieved_context[:3]:
            metadata = context_item["metadata"]
            source = metadata.get("source", "unknown source")
            page = metadata.get("page", None)

            if page is not None:
                page = int(page) + 1

            sources.append({
                "source": source,
                "page": page,
                "score": context_item["final_score"]
            })

        results.append({
            "id": question_id,
            "question": question,
            "answer": answer,
            "expected_keywords": expected_keywords,
            "evaluation": eval_result,
            "top_sources": sources
        })

    passed_count = sum(item["evaluation"]["passed"] for item in results)
    total_count = len(results)
    overall_score = passed_count / total_count if total_count > 0 else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": total_count,
        "passed_questions": passed_count,
        "failed_questions": total_count - passed_count,
        "overall_pass_rate": overall_score,
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions: {total_count}")
    print(f"Passed questions: {passed_count}")
    print(f"Failed questions: {total_count - passed_count}")
    print(f"Overall pass rate: {overall_score:.2f}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
