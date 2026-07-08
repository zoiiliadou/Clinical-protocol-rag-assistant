import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


CHROMA_PATH = Path("data/vector_store")
COLLECTION_NAME = "clinical_protocol"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


STOPWORDS = {
    "what", "are", "the", "for", "with", "and", "that", "this", "from",
    "how", "does", "into", "about", "which", "when", "where", "who",
    "why", "is", "be", "to", "of", "in", "on", "as", "a", "an",
    "please", "tell", "me"
}


IMPORTANT_TERMS = {
    "pregnant": ["pregnant", "pregnancy"],
    "breastfeeding": ["breastfeeding", "breastfeed"],
    "exclusion": ["exclusion", "excluded", "exclude"],
    "inclusion": ["inclusion", "included", "include"],
    "criteria": ["criteria", "criterion"],
    "study": ["study"],
    "design": ["design", "randomized", "placebo-controlled", "observer-blind", "dose-finding"],
    "dosing": ["dose", "dosing", "vaccination", "schedule", "intervention"],
    "schedule": ["schedule", "dose", "dosing", "vaccination", "visit"],
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer for BM25 lexical retrieval.
    Keeps medical/product terms such as BNT162b2 and protocol-like identifiers.
    """
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text)

    return [
        token for token in tokens
        if token not in STOPWORDS
    ]


def bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    Lightweight BM25 implementation without external dependencies.
    BM25 improves exact keyword/code matching while ChromaDB handles semantic similarity.
    """
    tokenized_docs = [tokenize_for_bm25(doc) for doc in documents]
    query_terms = list(set(tokenize_for_bm25(query)))

    number_of_docs = len(tokenized_docs)

    if number_of_docs == 0 or not query_terms:
        return [0.0 for _ in documents]

    doc_lengths = [len(doc_tokens) for doc_tokens in tokenized_docs]
    average_doc_length = sum(doc_lengths) / number_of_docs if number_of_docs > 0 else 0.0

    document_frequencies = Counter()

    for doc_tokens in tokenized_docs:
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            document_frequencies[term] += 1

    scores = []

    for doc_tokens, doc_length in zip(tokenized_docs, doc_lengths):
        term_counts = Counter(doc_tokens)
        score = 0.0

        for term in query_terms:
            if term not in term_counts:
                continue

            df = document_frequencies.get(term, 0)

            idf = math.log(
                1 + ((number_of_docs - df + 0.5) / (df + 0.5))
            )

            tf = term_counts[term]

            denominator = tf + k1 * (
                1 - b + b * (doc_length / average_doc_length)
            )

            if denominator > 0:
                score += idf * ((tf * (k1 + 1)) / denominator)

        scores.append(score)

    return scores


def clean_for_dedup(text: str) -> str:
    """
    Removes repeated protocol headers/footers and normalizes text
    so that near-duplicate chunks can be detected more easily.
    """
    text = normalize_text(text)

    patterns_to_remove = [
        r"pf-07302048 \(bnt162 rna-based covid-19 vaccines\)",
        r"protocol c4591001",
        r"pfizer confidential",
        r"ct02-gsop clinical protocol template phase 1 2 3 4.*?",
        r"approved.*?gmt\)",
        r"page \d+",
        r"tmf doc id.*?",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def token_set(text: str) -> Set[str]:
    cleaned = clean_for_dedup(text)
    words = re.findall(r"[a-zA-Z]{3,}", cleaned)
    return {word for word in words if word not in STOPWORDS}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = token_set(text_a)
    tokens_b = token_set(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    return intersection / union


def is_table_of_contents_like(text: str) -> bool:
    text_norm = normalize_text(text)

    toc_signals = [
        "list of tables",
        "list of figures",
        "table of contents",
        "appendix",
    ]

    dotted_lines = text.count(".....")

    if dotted_lines >= 2:
        return True

    return any(signal in text_norm for signal in toc_signals)


def extract_query_terms(query: str) -> List[str]:
    query_lower = normalize_text(query)
    words = re.findall(r"[a-zA-Z]{3,}", query_lower)

    expanded_terms = set()

    for word in words:
        if word not in STOPWORDS:
            expanded_terms.add(word)

    for key, variants in IMPORTANT_TERMS.items():
        if key in query_lower:
            expanded_terms.update(variants)

    return list(expanded_terms)


def keyword_score(query: str, text: str) -> float:
    query_norm = normalize_text(query)
    text_norm = normalize_text(text)
    query_terms = extract_query_terms(query)

    score = 0.0

    for term in query_terms:
        if term in text_norm:
            score += 1.0

    # Penalize table-of-contents-like chunks unless the user explicitly asks for contents.
    if is_table_of_contents_like(text) and "table of contents" not in query_norm:
        score -= 6.0

    # Eligibility / exclusion boosts
    if "pregnant" in query_norm and "women who are pregnant or breastfeeding" in text_norm:
        score += 20.0

    if "exclusion" in query_norm and "exclusion criteria" in text_norm:
        score += 10.0

    if "inclusion" in query_norm and "inclusion criteria" in text_norm:
        score += 10.0

    if "study population" in text_norm and ("inclusion" in query_norm or "exclusion" in query_norm):
        score += 4.0

    # Study design boosts
    if "study design" in query_norm:
        if "study design" in text_norm:
            score += 20.0

        design_phrases = [
            "randomized",
            "placebo-controlled",
            "observer-blind",
            "dose-finding",
            "phase 1/2/3",
            "phase 1/2",
            "vaccine candidate",
            "safety, tolerability, immunogenicity",
            "efficacy"
        ]

        phrase_hits = sum(1 for phrase in design_phrases if phrase in text_norm)
        score += phrase_hits * 3.0

        irrelevant_phrases = [
            "study start date",
            "site closure",
            "data quality assurance",
            "source data verification",
        ]

        if any(phrase in text_norm for phrase in irrelevant_phrases):
            score -= 10.0

    # Dosing schedule boosts
    if "dosing schedule" in query_norm or "dose schedule" in query_norm:
        dosing_phrases = [
            "2-dose",
            "separated by 21 days",
            "separated by 21 or 60 days",
            "dose 1",
            "dose 2",
            "vaccination 1",
            "vaccination 2",
            "study intervention",
            "19 to 23 days",
            "56 to 70 days",
            "administer study intervention"
        ]

        phrase_hits = sum(1 for phrase in dosing_phrases if phrase in text_norm)
        score += phrase_hits * 3.0

        if "window for visit 2" in text_norm:
            score += 5.0

    return max(score, 0.0)


def get_keyword_candidates(db: Chroma, query: str, limit: int = 30):
    """
    BM25 + rule-based keyword retrieval over all chunks.

    This complements ChromaDB semantic retrieval:
    - ChromaDB finds meaning-based matches.
    - BM25 finds exact terms, codes, drug names, protocol identifiers.
    - keyword_score adds domain-specific boosts for important clinical sections.
    """
    raw = db.get(include=["documents", "metadatas"])

    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    bm25_result_scores = bm25_scores(query, documents)

    scored = []

    for doc_text, metadata, bm25_score_value in zip(documents, metadatas, bm25_result_scores):
        rule_score = keyword_score(query, doc_text)

        # Weighted ensemble score: BM25 lexical score + domain-specific keyword score.
        combined_score = bm25_score_value + rule_score

        if combined_score > 0:
            scored.append((doc_text, metadata, combined_score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:limit]


def deduplicate_results(results: List[Dict], similarity_threshold: float = 0.82) -> List[Dict]:
    """
    Keeps the best-scoring result and removes near-duplicate chunks.
    This is useful because the PDF contains repeated protocol versions/amendments.
    """
    deduplicated = []

    for candidate in results:
        candidate_text = candidate["content"]

        is_duplicate = False

        for kept in deduplicated:
            kept_text = kept["content"]
            similarity = jaccard_similarity(candidate_text, kept_text)

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(candidate)

    return deduplicated


def main():
    parser = argparse.ArgumentParser(
        description="Professional hybrid search over the C4591001 clinical protocol."
    )
    parser.add_argument("query", type=str, help="Question or search query for the protocol.")
    parser.add_argument("--k", type=int, default=5, help="Number of final results to return.")
    parser.add_argument("--semantic-k", type=int, default=30, help="Number of semantic candidates.")
    args = parser.parse_args()

    if not CHROMA_PATH.exists():
        logger.error(f"Vector store not found: {CHROMA_PATH}")
        logger.error("Run python src/ingest.py first.")
        return

    logger.info(f"Loading embedding model locally: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True
        },
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info("Connecting to ChromaDB...")
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    logger.info(f"Executing hybrid BM25 + ChromaDB search for: {args.query}")

    semantic_results = db.similarity_search_with_score(args.query, k=args.semantic_k)

    combined: Dict[str, Dict] = {}

    for doc, distance in semantic_results:
        key = clean_for_dedup(doc.page_content)[:400]
        combined[key] = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "semantic_distance": distance,
            "keyword_score": keyword_score(args.query, doc.page_content),
        }

    keyword_candidates = get_keyword_candidates(db, args.query, limit=30)

    for text, metadata, lexical_score in keyword_candidates:
        key = clean_for_dedup(text)[:400]

        if key not in combined:
            combined[key] = {
                "content": text,
                "metadata": metadata,
                "semantic_distance": None,
                "keyword_score": lexical_score,
            }
        else:
            combined[key]["keyword_score"] = max(combined[key]["keyword_score"], lexical_score)

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
    final_results = final_results[:args.k]

    if not final_results:
        print("\nNo relevant results found.")
        return

    print("\n" + "=" * 80)
    print("HYBRID BM25 + CHROMADB SEARCH RESULTS")
    print("=" * 80)

    for i, item in enumerate(final_results, start=1):
        metadata = item["metadata"]
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", None)

        if page is not None:
            page_display = int(page) + 1
        else:
            page_display = "unknown"

        semantic_distance = item["semantic_distance"]

        if semantic_distance is None:
            semantic_display = "BM25/keyword-only"
        else:
            semantic_display = f"{semantic_distance:.4f}"

        print(f"\n--- Match {i} ---")
        print(f"Final score: {item['final_score']:.4f}")
        print(f"Semantic distance: {semantic_display}")
        print(f"Lexical/BM25 score: {item['keyword_score']:.4f}")
        print(f"Source: {source}")
        print(f"Page: {page_display}")
        print("-" * 80)
        print(item["content"])

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
