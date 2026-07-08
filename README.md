# Clinical Protocol RAG Assistant

Local AI/RAG assistant for clinical protocol question answering.

The system indexes a clinical trial protocol PDF and allows users to ask natural language questions about study design, eligibility criteria, dosing schedules, safety monitoring, adverse events, and follow-up procedures.

Local application: http://localhost:8501  
Protocol source: https://www.nejm.org/doi/suppl/10.1056/NEJMoa2034577/suppl_file/nejmoa2034577_protocol.pdf

This is a personal portfolio project and a local medical document exploration prototype. It is not intended for clinical decision-making.

## Key Features

- Hybrid RAG retrieval using ChromaDB semantic search and BM25 keyword search
- Evidence-based answers generated from retrieved protocol passages
- Page-level citations with source pages and retrieval scores
- Query intent detection
- Evidence confidence estimation
- Lightweight clinical entity extraction
- No-answer handling for unsupported questions
- Local SQLite audit logging
- FastAPI backend
- Streamlit frontend
- Evaluation and API test scripts

## Technology Stack

- Python
- FastAPI
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- BM25
- SQLite

## Protocol PDF

The protocol PDF is not included in this repository due to file size and licensing considerations.

To run the project locally, download the public protocol PDF from:

https://www.nejm.org/doi/suppl/10.1056/NEJMoa2034577/suppl_file/nejmoa2034577_protocol.pdf

Then place it at:

```text
data/raw/C4591001_protocol.pdf
