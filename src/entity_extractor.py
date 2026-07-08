import argparse
import json
import re
from typing import Dict, List


NO_ANSWER_MESSAGE = "I could not find enough evidence in the protocol to answer this question."


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_entity_text(entity: str) -> str:
    """
    Clean entity text extracted from noisy PDF passages.
    """
    entity = re.sub(r"\s+", " ", entity).strip()
    entity = entity.replace("–", "-")

    replacements = {
        "COVID": "COVID-19",
        "AEs": "AE",
        "SAEs": "SAE"
    }

    return replacements.get(entity, entity)


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    unique_items = []

    for item in items:
        cleaned = clean_entity_text(item)

        if not cleaned:
            continue

        key = cleaned.lower()

        if key not in seen:
            unique_items.append(cleaned)
            seen.add(key)

    return unique_items


def find_regex_matches(patterns: List[str], text: str, flags=re.IGNORECASE) -> List[str]:
    matches = []

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=flags):
            if isinstance(match, tuple):
                match = " ".join(part for part in match if part)

            matches.append(str(match).strip())

    return unique_preserve_order(matches)


def extract_clinical_entities(text: str) -> Dict[str, List[str]]:
    """
    Lightweight clinical/protocol entity extraction.

    This is not a heavy pretrained Medical NER model.
    It is a local, rule-based NLP layer designed for clinical protocol documents.
    """
    entities = {
        "vaccine_or_product": [],
        "disease_or_condition": [],
        "study_design": [],
        "eligibility": [],
        "visits_and_procedures": [],
        "safety": [],
        "time_windows": [],
        "data_and_measurements": []
    }

    product_patterns = [
        r"\bBNT162[a-zA-Z0-9\-]*\b",
        r"\bPF-07302048\b",
        r"\bBNT162 RNA-Based COVID-19 Vaccines?\b",
        r"\bRNA-Based COVID-19 Vaccines?\b"
    ]

    disease_patterns = [
        r"\bCOVID-19\b",
        r"\bSARS-CoV-2\b",
        r"\bcoronavirus\b",
        r"\bCOVID\b"
    ]

    study_design_patterns = [
        r"\bPhase\s*1/2/3\b",
        r"\bPhase\s*1/2\b",
        r"\brandomized\b",
        r"\bplacebo-controlled\b",
        r"\bobserver-blind\b",
        r"\bdose-finding\b",
        r"\bvaccine candidate[\w\-– ]*selection\b",
        r"\befficacy study\b"
    ]

    eligibility_patterns = [
        r"\binclusion criteria\b",
        r"\bexclusion criteria\b",
        r"\bpregnant\b",
        r"\bpregnancy\b",
        r"\bbreastfeeding\b",
        r"\bimmunocompromised\b",
        r"\bautoimmune disease\b",
        r"\bhealthy adults?\b",
        r"\bmale or female participants?\b"
    ]

    visit_patterns = [
        r"\bVisit\s*\d+\b",
        r"\bVaccination\s*\d+\b",
        r"\bDay\s*\d+\b",
        r"\bFollow-up Visit\b",
        r"\bSafety Telephone Contact\b",
        r"\be-diary\b",
        r"\bstudy intervention\b",
        r"\bconvalescent visit\b",
        r"\bCOVID-19 illness visit\b"
    ]

    safety_patterns = [
        r"\bAE\b",
        r"\bAEs\b",
        r"\bSAE\b",
        r"\bSAEs\b",
        r"\badverse events?\b",
        r"\bserious adverse events?\b",
        r"\bsafety monitoring\b",
        r"\bhospitalization\b",
        r"\blife-threatening\b",
        r"\bdeath\b"
    ]

    time_patterns = [
        r"\b\d+\s*to\s*\d+\s*days?\b",
        r"\b\d+\s*-\s*\d+\s*days?\b",
        r"\b\d+\s*to\s*\d+\s*months?\b",
        r"\b\d+\s*-\s*\d+\s*months?\b",
        r"\bDay\s*\d+\b",
        r"\b1-Month\b",
        r"\b6-Month\b",
        r"\b12-Month\b",
        r"\b24-Month\b"
    ]

    measurement_patterns = [
        r"\blaboratory tests?\b",
        r"\bimmunogenicity\b",
        r"\bNAAT\b",
        r"\bserological parameters?\b",
        r"\bantibody levels?\b",
        r"\bblood samples?\b",
        r"\bnasal swab\b",
        r"\bCRFs?\b",
        r"\bsource documents?\b"
    ]

    entities["vaccine_or_product"] = find_regex_matches(product_patterns, text)
    entities["disease_or_condition"] = find_regex_matches(disease_patterns, text)
    entities["study_design"] = find_regex_matches(study_design_patterns, text)
    entities["eligibility"] = find_regex_matches(eligibility_patterns, text)
    entities["visits_and_procedures"] = find_regex_matches(visit_patterns, text)
    entities["safety"] = find_regex_matches(safety_patterns, text)
    entities["time_windows"] = find_regex_matches(time_patterns, text)
    entities["data_and_measurements"] = find_regex_matches(measurement_patterns, text)

    return entities


def flatten_entities(entities: Dict[str, List[str]]) -> List[str]:
    all_entities = []

    for values in entities.values():
        all_entities.extend(values)

    return unique_preserve_order(all_entities)


def contains_any_term(text: str, terms: List[str]) -> bool:
    """
    Match complete terms instead of raw substrings.
    This avoids false positives such as matching 'fee' inside 'breastfeeding'.
    """
    text_norm = normalize_text(text)

    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"

        if re.search(pattern, text_norm):
            return True

    return False


def classify_query_intent(question: str) -> str:
    """
    Lightweight query intent classification for clinical protocol questions.
    """
    question_norm = normalize_text(question)

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

    eligibility_terms = [
        "inclusion",
        "exclusion",
        "eligible",
        "eligibility",
        "pregnant",
        "pregnancy",
        "breastfeeding"
    ]

    dosing_visit_terms = [
        "dose",
        "dosing",
        "vaccination",
        "visit",
        "schedule",
        "follow-up",
        "intervention"
    ]

    safety_terms = [
        "adverse event",
        "serious adverse",
        "sae",
        "ae",
        "safety",
        "hospitalization"
    ]

    outcome_data_terms = [
        "endpoint",
        "objective",
        "outcome",
        "immunogenicity",
        "laboratory",
        "sample",
        "data"
    ]

    study_design_terms = [
        "study design",
        "randomized",
        "placebo",
        "observer-blind",
        "phase"
    ]

    if contains_any_term(question_norm, financial_terms):
        return "unsupported_administrative_financial"

    if contains_any_term(question_norm, eligibility_terms):
        return "eligibility"

    if contains_any_term(question_norm, dosing_visit_terms):
        return "dosing_and_visits"

    if contains_any_term(question_norm, safety_terms):
        return "safety"

    if contains_any_term(question_norm, outcome_data_terms):
        return "outcomes_and_data"

    if contains_any_term(question_norm, study_design_terms):
        return "study_design"

    return "general_protocol_search"


def estimate_evidence_confidence(question: str, answer: str, retrieved_results: List[Dict]) -> Dict[str, str]:
    """
    Estimate evidence confidence using simple interpretable signals:
    - whether the answer is a no-answer response
    - whether retrieved evidence exists
    - top retrieval score
    - query/evidence lexical overlap

    This is not a probability. It is an interpretable confidence label for the UI.
    """
    if NO_ANSWER_MESSAGE.lower() in answer.lower():
        return {
            "label": "Low",
            "reason": "The system did not find sufficient supporting evidence in the indexed protocol."
        }

    if not retrieved_results:
        return {
            "label": "Low",
            "reason": "No retrieved evidence passages were available."
        }

    top_score = float(retrieved_results[0].get("final_score", 0.0))

    evidence_text = " ".join(
        item.get("content", "")
        for item in retrieved_results[:3]
    )

    question_terms = set(
        term for term in re.findall(r"[a-zA-Z]{3,}", normalize_text(question))
        if term not in {
            "what", "are", "the", "for", "with", "and", "that", "this",
            "from", "how", "does", "into", "about", "which", "when",
            "where", "who", "why"
        }
    )

    evidence_norm = normalize_text(evidence_text)

    if question_terms:
        matched_terms = [term for term in question_terms if term in evidence_norm]
        overlap = len(matched_terms) / len(question_terms)
    else:
        overlap = 0.0

    if top_score >= 20 and overlap >= 0.50:
        return {
            "label": "High",
            "reason": "The answer is supported by high-scoring retrieved evidence with strong query-term overlap."
        }

    if top_score >= 8 and overlap >= 0.30:
        return {
            "label": "Medium",
            "reason": "The answer is supported by retrieved evidence, but the evidence match is moderate."
        }

    return {
        "label": "Low",
        "reason": "The retrieved evidence is weak or only partially related to the question."
    }


def analyze_question_and_evidence(question: str, answer: str, retrieved_results: List[Dict]) -> Dict:
    intent = classify_query_intent(question)
    confidence = estimate_evidence_confidence(question, answer, retrieved_results)

    if NO_ANSWER_MESSAGE.lower() in answer.lower():
        entities = {
            "vaccine_or_product": [],
            "disease_or_condition": [],
            "study_design": [],
            "eligibility": [],
            "visits_and_procedures": [],
            "safety": [],
            "time_windows": [],
            "data_and_measurements": []
        }
    else:
        evidence_text = " ".join(
            item.get("content", "")
            for item in retrieved_results[:5]
        )

        entities = extract_clinical_entities(evidence_text)

    return {
        "query_intent": intent,
        "evidence_confidence": confidence,
        "entities": entities,
        "entity_count": len(flatten_entities(entities))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight clinical entity extraction and intent detection."
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text or question to analyze."
    )

    args = parser.parse_args()

    entities = extract_clinical_entities(args.text)
    intent = classify_query_intent(args.text)

    output = {
        "query_intent": intent,
        "entities": entities,
        "entity_count": len(flatten_entities(entities))
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
