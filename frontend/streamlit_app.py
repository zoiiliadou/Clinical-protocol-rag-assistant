from html import escape

import requests
import streamlit as st
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL}/ask"
HEALTH_URL = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="Clinical Protocol RAG Assistant",
    layout="wide"
)


st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8fafc;
            color: #111827;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 2.4rem;
            padding-bottom: 2.2rem;
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", Arial, sans-serif;
        }

        h1 {
            color: #0f172a;
            font-size: 2.35rem !important;
            font-weight: 760 !important;
            letter-spacing: -0.02em;
        }

        h2 {
            color: #0f172a;
            font-size: 1.85rem !important;
            font-weight: 720 !important;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }

        h3 {
            color: #0f172a;
            font-size: 1.28rem !important;
            font-weight: 680 !important;
        }

        p, label, div, span {
            font-size: 16px;
        }

        section[data-testid="stSidebar"] {
            background-color: #f1f5f9;
            border-right: 1px solid #dbe3ea;
        }

        section[data-testid="stSidebar"] h2 {
            font-size: 1.35rem !important;
            font-weight: 720 !important;
            color: #0f172a;
            margin-top: 1.2rem;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] span {
            font-size: 15.5px;
        }

        .hero-card {
            background-color: #ffffff;
            border: 1px solid #dbe3ea;
            border-radius: 16px;
            padding: 34px 36px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
        }

        .hero-title {
            font-size: 38px;
            font-weight: 780;
            color: #0f172a;
            margin-bottom: 12px;
            letter-spacing: -0.025em;
        }

        .hero-subtitle {
            font-size: 17px;
            color: #475569;
            line-height: 1.7;
            max-width: 980px;
        }

        .badge {
            display: inline-block;
            background-color: #f1f5f9;
            color: #334155;
            padding: 7px 13px;
            border-radius: 999px;
            font-size: 13.5px;
            margin-right: 7px;
            margin-top: 16px;
            border: 1px solid #dbe3ea;
        }

        .notice {
            background-color: #ffffff;
            border: 1px solid #dbe3ea;
            border-left: 5px solid #64748b;
            border-radius: 13px;
            padding: 16px 18px;
            color: #334155;
            line-height: 1.6;
            margin-bottom: 20px;
            font-size: 16px;
        }

        .status-ok {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 5px solid #047857;
            border-radius: 13px;
            padding: 15px 18px;
            color: #065f46;
            margin-bottom: 22px;
            font-size: 16px;
        }

        .status-error {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 5px solid #991b1b;
            border-radius: 13px;
            padding: 15px 18px;
            color: #7f1d1d;
            margin-bottom: 22px;
            font-size: 16px;
        }

        .answer-box {
            background-color: #ffffff;
            border: 1px solid #dbe3ea;
            border-left: 5px solid #0f766e;
            border-radius: 13px;
            padding: 20px;
            font-size: 17px;
            line-height: 1.75;
            color: #1e293b;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
        }

        .analysis-card {
            background-color: #ffffff;
            border: 1px solid #dbe3ea;
            border-radius: 13px;
            padding: 17px 18px;
            min-height: 118px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
        }

        .analysis-label {
            font-size: 13px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            margin-bottom: 8px;
            font-weight: 650;
        }

        .analysis-value {
            font-size: 18px;
            color: #0f172a;
            font-weight: 720;
            margin-bottom: 6px;
        }

        .analysis-reason {
            font-size: 14px;
            color: #475569;
            line-height: 1.55;
        }

        .confidence-high {
            color: #047857;
        }

        .confidence-medium {
            color: #b45309;
        }

        .confidence-low {
            color: #991b1b;
        }

        .entity-section {
            background-color: #ffffff;
            border: 1px solid #dbe3ea;
            border-radius: 13px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }

        .entity-category {
            font-size: 14px;
            color: #334155;
            font-weight: 720;
            margin-bottom: 8px;
        }

        .entity-chip {
            display: inline-block;
            background-color: #f1f5f9;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 6px 10px;
            margin: 3px 4px 5px 0;
            font-size: 13.5px;
        }

        .source-card {
            background-color: #ffffff;
            border: 1px solid #dbe3ea;
            border-radius: 13px;
            padding: 15px 17px;
            margin-bottom: 11px;
        }

        .source-title {
            font-weight: 720;
            color: #0f172a;
            font-size: 16px;
        }

        .source-meta {
            color: #64748b;
            font-size: 14px;
            margin-top: 4px;
        }

        .small-muted {
            color: #64748b;
            font-size: 14px;
            line-height: 1.55;
        }

        .section-caption {
            color: #64748b;
            font-size: 15px;
            margin-top: -6px;
            margin-bottom: 16px;
        }

        div.stButton > button {
            border-radius: 9px;
            border: 1px solid #cbd5e1;
            background-color: #ffffff;
            color: #1e293b;
            font-size: 15.5px;
            padding: 0.55rem 0.95rem;
            min-height: 42px;
        }

        div.stButton > button:hover {
            border-color: #334155;
            color: #0f172a;
            background-color: #f8fafc;
        }

        div.stButton > button[kind="primary"] {
            background-color: #111827;
            color: #ffffff;
            border: 1px solid #111827;
            font-weight: 650;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #1f2937;
            color: #ffffff;
            border: 1px solid #1f2937;
        }

        .stTabs [data-baseweb="tab"] {
            color: #334155;
            font-size: 15px;
            padding-top: 10px;
            padding-bottom: 10px;
        }

        .stTabs [aria-selected="true"] {
            color: #0f172a;
            font-weight: 720;
        }

        textarea {
            font-size: 16px !important;
            line-height: 1.55 !important;
        }

        input, select {
            font-size: 16px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def check_backend_health():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None

    return None


def ask_backend(question: str, role: str, k: int):
    response = requests.post(
        API_URL,
        json={
            "question": question,
            "role": role,
            "k": k
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def readable_label(text: str) -> str:
    return text.replace("_", " ").title()


def confidence_class(label: str) -> str:
    label_lower = label.lower()

    if label_lower == "high":
        return "confidence-high"

    if label_lower == "medium":
        return "confidence-medium"

    return "confidence-low"


def render_analysis(analysis: dict):
    intent = analysis.get("query_intent", "unknown")
    confidence = analysis.get("evidence_confidence", {})
    confidence_label = confidence.get("label", "Unknown")
    confidence_reason = confidence.get("reason", "No confidence reason available.")
    entity_count = analysis.get("entity_count", 0)

    confidence_css = confidence_class(confidence_label)

    st.markdown("## AI / NLP Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="analysis-card">
                <div class="analysis-label">Detected Query Intent</div>
                <div class="analysis-value">{escape(readable_label(intent))}</div>
                <div class="analysis-reason">
                    Lightweight intent classification based on clinical protocol terminology.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="analysis-card">
                <div class="analysis-label">Evidence Confidence</div>
                <div class="analysis-value {confidence_css}">{escape(confidence_label)}</div>
                <div class="analysis-reason">{escape(confidence_reason)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="analysis-card">
                <div class="analysis-label">Detected Entities</div>
                <div class="analysis-value">{entity_count}</div>
                <div class="analysis-reason">
                    Rule-based clinical entity extraction from retrieved evidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    entities = analysis.get("entities", {})

    st.markdown("### Detected Clinical Entities")

    has_entities = False

    for category, values in entities.items():
        if not values:
            continue

        has_entities = True

        chips = "".join(
            f'<span class="entity-chip">{escape(str(value))}</span>'
            for value in values[:12]
        )

        extra_count = max(0, len(values) - 12)

        if extra_count > 0:
            chips += f'<span class="entity-chip">+{extra_count} more</span>'

        st.markdown(
            f"""
            <div class="entity-section">
                <div class="entity-category">{escape(readable_label(category))}</div>
                {chips}
            </div>
            """,
            unsafe_allow_html=True
        )

    if not has_entities:
        st.markdown(
            """
            <div class="entity-section">
                <div class="analysis-reason">No clinical entities were detected for this answer.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


with st.sidebar:
    st.markdown("## Indexed Document")

    with st.container(border=True):
        st.markdown("**Document**")
        st.write("C4591001_protocol.pdf")

        st.markdown("**Document type**")
        st.write("Clinical trial protocol")

        st.markdown("**Pages indexed**")
        st.write("376")

        st.markdown("**Topic**")
        st.write("BNT162 RNA-Based COVID-19 vaccine clinical trial")

    st.markdown("## Protocol Coverage")

    with st.container(border=True):
        st.write("Study design")
        st.write("Objectives and endpoints")
        st.write("Eligibility criteria")
        st.write("Inclusion / exclusion criteria")
        st.write("Dosing and visit schedule")
        st.write("Randomization and blinding")
        st.write("Safety monitoring")
        st.write("Adverse events")
        st.write("Follow-up procedures")
        st.write("Discontinuation / withdrawal")
        st.write("Laboratory and immunogenicity samples")

    st.markdown("## System Mode")

    with st.container(border=True):
        st.write("Local execution")
        st.write("Hybrid retrieval")
        st.write("ChromaDB + BM25")
        st.write("AI/NLP analysis layer")
        st.write("FastAPI backend")
        st.write("SQLite audit logs")


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Clinical Protocol RAG Assistant</div>
        <div class="hero-subtitle">
            A local document question-answering system for an indexed clinical trial protocol.
            The assistant retrieves relevant evidence from the protocol and returns concise answers
            with page-level citations, intent detection, evidence confidence, and clinical entity extraction.
        </div>
        <span class="badge">Local RAG</span>
        <span class="badge">Hybrid retrieval</span>
        <span class="badge">ChromaDB + BM25</span>
        <span class="badge">Evidence confidence</span>
        <span class="badge">Clinical entities</span>
        <span class="badge">Audit logging</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="notice">
        This local demo is designed for medical document retrieval and protocol exploration.
        It is not a medical decision-making tool. Questions should refer to the indexed protocol shown in the sidebar.
    </div>
    """,
    unsafe_allow_html=True
)


health = check_backend_health()

if health and health.get("vector_store_loaded"):
    st.markdown(
        """
        <div class="status-ok">
            Backend is running and the vector store is loaded.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="status-error">
            FastAPI backend is not running or the vector store is not loaded.
            Start it with: <b>uvicorn app.api.main:app --reload</b>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("## Ask a question about the indexed protocol")
st.markdown(
    """
    <div class="section-caption">
        Select an example question or type any question related to the indexed clinical trial protocol.
    </div>
    """,
    unsafe_allow_html=True
)


example_questions = {
    "Overview": {
        "Study design": "What is the study design?",
        "Study objectives": "What are the study objectives?",
        "Study endpoints": "What are the study endpoints?",
        "End of study": "How is the end of the study defined?"
    },
    "Eligibility": {
        "Inclusion criteria": "What are the inclusion criteria for participants?",
        "Exclusion criteria": "What are the exclusion criteria?",
        "Pregnancy exclusion": "What are the exclusion criteria for pregnant women?",
        "Temporary delay criteria": "What are the temporary delay criteria?"
    },
    "Intervention and Visits": {
        "Dosing schedule": "What is the dosing schedule?",
        "Vaccination visits": "What procedures are performed at vaccination visits?",
        "Follow-up schedule": "What is the follow-up schedule?",
        "Study intervention": "What study intervention is administered?"
    },
    "Safety": {
        "Adverse events": "What adverse events are collected?",
        "Serious adverse events": "How are serious adverse events collected?",
        "Safety monitoring": "What safety monitoring procedures are described?",
        "Pregnancy during study": "What happens if a participant becomes pregnant?"
    },
    "Data and Procedures": {
        "Laboratory tests": "What laboratory tests are described in the protocol?",
        "Immunogenicity samples": "What immunogenicity samples are collected?",
        "COVID-19 illness visit": "What happens during a potential COVID-19 illness visit?",
        "Convalescent visit": "What happens during the convalescent visit?"
    },
    "Discontinuation": {
        "Withdrawal": "What happens if a participant withdraws from the study?",
        "Discontinuation criteria": "What are the reasons for discontinuation of study intervention?",
        "Protocol deviation": "How are protocol deviations described?",
        "Missed visit": "What happens if a participant misses a study visit?"
    }
}


if "question" not in st.session_state:
    st.session_state.question = "What is the study design?"


tabs = st.tabs(list(example_questions.keys()))

for tab, category in zip(tabs, example_questions.keys()):
    with tab:
        cols = st.columns(4)

        for index, (label, question_text) in enumerate(example_questions[category].items()):
            with cols[index % 4]:
                if st.button(label, key=f"{category}_{label}"):
                    st.session_state.question = question_text


question = st.text_area(
    "Question",
    value=st.session_state.question,
    height=135,
    help="Ask any question related to the indexed clinical trial protocol."
)

col1, col2 = st.columns([1, 2])

with col1:
    role = st.selectbox(
        "User role",
        ["researcher", "clinician", "reviewer"],
        help="The selected role is stored in the local audit log."
    )

with col2:
    k = st.slider(
        "Evidence passages to retrieve",
        min_value=3,
        max_value=10,
        value=5,
        help="Higher values retrieve more supporting passages, but may include repeated evidence."
    )


if st.button("Ask", type="primary"):
    if not question.strip():
        st.error("Please enter a question.")
    elif not health:
        st.error("Cannot connect to the FastAPI backend.")
    else:
        with st.spinner("Retrieving evidence and running AI/NLP analysis..."):
            try:
                data = ask_backend(question, role, k)

                st.markdown("## Answer")
                st.markdown(
                    f"""
                    <div class="answer-box">
                        {escape(data["answer"])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                analysis = data.get("analysis", {})
                render_analysis(analysis)

                st.markdown("## Sources")

                sources = data.get("sources", [])

                if sources:
                    for source in sources:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-title">{escape(source["source"])}, page {source["page"]}</div>
                                <div class="source-meta">Retrieval score: {source["score"]}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No sources returned.")

                with st.expander("Developer details"):
                    st.json(data)

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")


st.divider()

st.markdown(
    """
    <span class="small-muted">
    Privacy note: The current version runs locally. The indexed document, embeddings,
    vector database, generated answers, and audit logs remain on the local machine.
    </span>
    """,
    unsafe_allow_html=True
)
