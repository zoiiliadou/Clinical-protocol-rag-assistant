import requests


BASE_URL = "http://127.0.0.1:8000"


def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health", timeout=10)

    assert response.status_code == 200, "Health endpoint did not return 200."

    data = response.json()

    assert data.get("status") == "ok", "Backend status is not ok."
    assert data.get("vector_store_loaded") is True, "Vector store is not loaded."

    print("Health endpoint passed.")


def test_ask_endpoint_supported_question():
    payload = {
        "question": "What is the study design?",
        "role": "researcher",
        "k": 5
    }

    response = requests.post(f"{BASE_URL}/ask", json=payload, timeout=60)

    assert response.status_code == 200, "Ask endpoint did not return 200."

    data = response.json()

    answer = data.get("answer", "").lower()
    sources = data.get("sources", [])

    assert "randomized" in answer, "Expected keyword 'randomized' not found in answer."
    assert "placebo-controlled" in answer, "Expected keyword 'placebo-controlled' not found in answer."
    assert len(sources) > 0, "No sources returned."

    print("Ask endpoint supported-question test passed.")


def test_ask_endpoint_no_answer_question():
    payload = {
        "question": "What is the hospital admission cost?",
        "role": "researcher",
        "k": 5
    }

    response = requests.post(f"{BASE_URL}/ask", json=payload, timeout=60)

    assert response.status_code == 200, "Ask endpoint did not return 200."

    data = response.json()

    answer = data.get("answer", "").lower()

    assert "could not find enough evidence" in answer, "No-answer behavior did not trigger."

    print("Ask endpoint no-answer test passed.")


def main():
    print("=" * 80)
    print("FASTAPI BACKEND TESTS")
    print("=" * 80)

    test_health_endpoint()
    test_ask_endpoint_supported_question()
    test_ask_endpoint_no_answer_question()

    print("=" * 80)
    print("All API tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
