import argparse
import logging
import re
from entity_extractor import analyze_question_and_evidence
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from audit_logger import log_query

from query import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    keyword_score,
    get_keyword_candidates,
    clean_for_dedup,
    deduplicate_results,
    normalize_text,
    extract_query_terms,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_vector_store() -> Chroma:
    """
    Load the local ChromaDB vector store and the local embedding model.
    No new model is downloaded because local_files_only=True is used.
    """
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PATH}. Run python src/ingest.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True
        },
        encode_kwargs={"normalize_embeddings": True}
    )

    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return db


def retrieve_context(db: Chroma, query: str, final_k: int = 5, semantic_k: int = 30) -> List[Dict]:
    """
    Retrieve evidence using the same hybrid strategy as query.py:
    semantic retrieval + keyword candidates + reranking + deduplication.
    """
    semantic_results = db.similarity_search_with_score(query, k=semantic_k)

    combined: Dict[str, Dict] = {}

    for doc, distance in semantic_results:
        key = clean_for_dedup(doc.page_content)[:400]
        combined[key] = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "semantic_distance": distance,
            "keyword_score": keyword_score(query, doc.page_content),
        }

    keyword_candidates = get_keyword_candidates(db, query, limit=30)

    for text, metadata, kscore in keyword_candidates:
        key = clean_for_dedup(text)[:400]

        if key not in combined:
            combined[key] = {
                "content": text,
                "metadata": metadata,
                "semantic_distance": None,
                "keyword_score": kscore,
            }
        else:
            combined[key]["keyword_score"] = max(combined[key]["keyword_score"], kscore)

    final_results = []

    for item in combined.values():
        semantic_distance = item["semantic_distance"]

        if semantic_distance is None:
            semantic_component = 0.0
        else:
            semantic_component = max(0.0, 2.0 - float(semantic_distance))

        final_score = semantic_component + item["keyword_score"]

        final_results.append({
            **item,
            "final_score": final_score
        })

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    final_results = deduplicate_results(final_results)

    return final_results[:final_k]


def split_into_sentences(text: str) -> List[str]:
    """
    Split retrieved text into readable candidate sentences.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


def sentence_relevance_score(query: str, sentence: str) -> float:
    """
    Score a sentence against the user query using query terms and keyword scoring.
    """
    query_terms = extract_query_terms(query)
    sentence_norm = normalize_text(sentence)

    score = 0.0

    for term in query_terms:
        if term in sentence_norm:
            score += 1.0

    score += keyword_score(query, sentence)

    return score


def is_relevant_source_for_query(query: str, text: str) -> bool:
    """
    Filter evidence chunks used as sources so the final answer does not cite
    irrelevant sections such as discontinuation when the user asks about inclusion.
    """
    query_norm = normalize_text(query)
    text_norm = normalize_text(text)

    if "pregnant" in query_norm:
        return "women who are pregnant or breastfeeding" in text_norm

    if "inclusion" in query_norm and "criteria" in query_norm:
        return (
            "5.1. inclusion criteria" in text_norm
            or "participants are eligible to be included in the study" in text_norm
            or "male or female participants" in text_norm
            or "female participant reproductive inclusion criteria" in text_norm
            or "male participant reproductive inclusion criteria" in text_norm
        )

    if "exclusion" in query_norm and "criteria" in query_norm:
        return (
            "5.2. exclusion criteria" in text_norm
            or "participants are excluded from the study" in text_norm
            or "women who are pregnant or breastfeeding" in text_norm
        )

    if "study design" in query_norm:
        return (
            "study design" in text_norm
            or (
                "randomized" in text_norm
                and "placebo-controlled" in text_norm
                and "observer-blind" in text_norm
            )
        )

    if "dosing schedule" in query_norm or "dose schedule" in query_norm:
        return (
            "vaccination 1" in text_norm
            or "vaccination 2" in text_norm
            or "19 to 23 days" in text_norm
            or "56 to 70 days" in text_norm
            or "administer study intervention" in text_norm
        )

    return True


def filtered_results_for_output(query: str, results: List[Dict]) -> List[Dict]:
    """
    Keep query-relevant evidence for the final displayed sources/evidence.
    If filtering becomes too strict, fall back to the original results.
    """
    filtered = [
        item for item in results
        if is_relevant_source_for_query(query, item["content"])
    ]

    if filtered:
        return filtered

    return results


def unique_sources(query: str, results: List[Dict], max_sources: int = 3) -> List[Tuple[str, int]]:
    """
    Return unique source-page pairs for the final answer.
    """
    relevant_results = filtered_results_for_output(query, results)

    seen = set()
    sources = []

    for item in relevant_results:
        metadata = item["metadata"]
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", None)

        if page is None:
            page_display = -1
        else:
            page_display = int(page) + 1

        key = (source, page_display)

        if key not in seen:
            seen.add(key)
            sources.append(key)

        if len(sources) >= max_sources:
            break

    return sources


def generate_extractive_answer(query: str, results: List[Dict]) -> str:
    """
    Generate a concise evidence-based answer without using an external LLM.
    This is intentionally extractive/rule-based to keep the system local and lightweight.
    """
    no_answer_message = (
        "I could not find enough evidence in the protocol to answer this question."
    )

    if not results:
        return no_answer_message

    query_norm = normalize_text(query)
    top_text = "\n".join(item["content"] for item in results[:3])
    top_text_norm = normalize_text(top_text)

    # High-confidence clinical eligibility answer: pregnancy / breastfeeding.
    # This must run before the generic no-answer guard, because the question may contain
    # broad words such as "criteria" that are not always repeated in the exact evidence sentence.
    if (
        "pregnant" in query_norm
        and (
            "women who are pregnant or breastfeeding" in top_text_norm
            or ("pregnant" in top_text_norm and "breastfeeding" in top_text_norm)
        )
    ):
        return "Women who are pregnant or breastfeeding are excluded from the study."

    # No-answer guard for financial / administrative questions.
    # The protocol may mention hospital admission as a safety event,
    # but it does not necessarily contain hospital costs, billing, or prices.
    financial_terms = [
        "cost",
        "price",
        "payment",
        "billing",
        "insurance",
        "reimbursement",
        "fee",
        "charge",
        "charges"
    ]

    if any(term in query_norm for term in financial_terms):
        if not any(term in top_text_norm for term in financial_terms):
            return no_answer_message

    # Inclusion criteria answer
    if "inclusion" in query_norm and "criteria" in query_norm:
        for item in results:
            text_norm = normalize_text(item["content"])

            if (
                "5.1. inclusion criteria" in text_norm
                or "participants are eligible to be included in the study" in text_norm
            ):
                if "male or female participants" in text_norm:
                    return (
                        "Participants are eligible only if all inclusion criteria apply. "
                        "The protocol states that eligible participants include male or female "
                        "participants within the specified age ranges, depending on the study stage. "
                        "Participants must also be willing and able to comply with scheduled visits, "
                        "the vaccination plan, laboratory tests, lifestyle considerations, and other "
                        "study procedures."
                    )

                return (
                    "Participants are eligible only if all inclusion criteria listed in the protocol "
                    "apply. The cited protocol section contains the full inclusion criteria."
                )

    # Study design answer
    if "study design" in query_norm:
        for item in results:
            sentences = split_into_sentences(item["content"])

            for sentence in sentences:
                sentence_norm = normalize_text(sentence)

                if (
                    "randomized" in sentence_norm
                    and "placebo-controlled" in sentence_norm
                    and "observer-blind" in sentence_norm
                ):
                    cleaned_sentence = re.sub(
                        r"^(overall design\s*)",
                        "",
                        sentence,
                        flags=re.IGNORECASE
                    ).strip()

                    return cleaned_sentence

    # Dosing schedule answer
    if "dosing schedule" in query_norm or "dose schedule" in query_norm:
        if "19 to 23 days" in top_text_norm and "56 to 70 days" in top_text_norm:
            return (
                "The protocol describes Vaccination 1 at Visit 1 / Day 1 and Vaccination 2 "
                "at Visit 2, either 19 to 23 days or 56 to 70 days after Visit 1, depending "
                "on the assigned dosing schedule."
            )

        if "19 to 23 days" in top_text_norm:
            return (
                "The protocol describes Vaccination 1 at Visit 1 / Day 1 and Vaccination 2 "
                "at Visit 2, 19 to 23 days after Visit 1."
            )

    # Generic no-answer guard for weak evidence.
    # This runs after the high-confidence protocol-specific answers.
    query_terms = [
        term for term in extract_query_terms(query)
        if term not in {
            "what", "are", "the", "for", "with", "and", "that", "this",
            "from", "how", "does", "into", "about", "which", "when",
            "where", "who", "why", "study", "protocol", "criteria"
        }
    ]

    if query_terms:
        matched_terms = [
            term for term in query_terms
            if term in top_text_norm
        ]

        coverage = len(matched_terms) / len(query_terms)

        if coverage < 0.30:
            return no_answer_message

    # Generic extractive fallback
    candidate_sentences = []

    for item in results[:5]:
        text_norm = normalize_text(item["content"])

        # Avoid irrelevant discontinuation chunks for eligibility questions
        if (
            ("inclusion" in query_norm or "exclusion" in query_norm)
            and "discontinuation of study intervention" in text_norm
        ):
            continue

        for sentence in split_into_sentences(item["content"]):
            score = sentence_relevance_score(query, sentence)

            if score > 0:
                candidate_sentences.append((sentence, score))

    candidate_sentences.sort(key=lambda x: x[1], reverse=True)

    selected = []
    seen = set()

    for sentence, _ in candidate_sentences:
        cleaned = normalize_text(sentence)

        if cleaned not in seen:
            selected.append(sentence)
            seen.add(cleaned)

        if len(selected) >= 3:
            break

    if not selected:
        return no_answer_message

    return " ".join(selected)
    

def print_answer(query: str, answer: str, results: List[Dict]) -> None:
    """
    Print a clean answer, sources, and top evidence.
    """
    relevant_results = filtered_results_for_output(query, results)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    sources = unique_sources(query, results)

    for source, page in sources:
        if page == -1:
            print(f"- {source}, page unknown")
        else:
            print(f"- {source}, page {page}")

    print("\n" + "=" * 80)
    print("TOP EVIDENCE")
    print("=" * 80)

    for i, item in enumerate(relevant_results[:3], start=1):
        metadata = item["metadata"]
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", None)
        page_display = int(page) + 1 if page is not None else "unknown"

        print(f"\n--- Evidence {i} ---")
        print(f"Source: {source}")
        print(f"Page: {page_display}")
        print(f"Score: {item['final_score']:.4f}")
        print("-" * 80)
        print(item["content"][:1200])


def main():
    parser = argparse.ArgumentParser(
        description="Evidence-based Q&A over the C4591001 clinical protocol."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Question to ask over the protocol."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved evidence chunks."
    )

    parser.add_argument(
        "--role",
        type=str,
        default="researcher",
        help="User role for audit logging, e.g. researcher, clinician, reviewer."
    )

    args = parser.parse_args()

    question = args.query

    logger.info("Loading vector store...")
    db = load_vector_store()

    logger.info(f"Retrieving evidence for: {question}")
    results = retrieve_context(db, question, final_k=args.k)

    answer = generate_extractive_answer(question, results)

    analysis = analyze_question_and_evidence(
        question=question,
        answer=answer,
        retrieved_results=results
    )

    audit_sources = []

    for item in results[:3]:
        metadata = item.get("metadata", {})
        audit_sources.append({
            "source": metadata.get("source", "unknown source"),
            "page": metadata.get("page", None),
            "score": item.get("final_score", 0.0)
        })

    log_query(
        question=question,
        answer=answer,
        sources=audit_sources,
        user_role=args.role,
        status="success"
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for item in results[:3]:
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", None)

        if page is not None:
            page_display = int(page) + 1
        else:
            page_display = "unknown"

        print(f"- {source}, page {page_display}")

    print("\n" + "=" * 80)
    print("AI / NLP ANALYSIS")
    print("=" * 80)

    print(f"Query intent: {analysis['query_intent']}")

    confidence = analysis["evidence_confidence"]
    print(f"Evidence confidence: {confidence['label']}")
    print(f"Confidence reason: {confidence['reason']}")

    print("\nDetected clinical entities:")

    entities = analysis["entities"]
    has_entities = False

    for category, values in entities.items():
        if values:
            has_entities = True
            readable_category = category.replace("_", " ").title()
            print(f"- {readable_category}: {', '.join(values)}")

    if not has_entities:
        print("- None detected")

    print("\n" + "=" * 80)
    print("TOP EVIDENCE")
    print("=" * 80)

    for i, item in enumerate(results[:3], start=1):
        metadata = item.get("metadata", {})
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", None)

        if page is not None:
            page_display = int(page) + 1
        else:
            page_display = "unknown"

        print(f"\n--- Evidence {i} ---")
        print(f"Source: {source}")
        print(f"Page: {page_display}")
        print(f"Score: {item.get('final_score', 0.0):.4f}")
        print("-" * 80)
        print(item["content"])


if __name__ == "__main__":
    main()
