# Clinical Protocol RAG Assistant

Local Application: http://localhost:8501

A personal portfolio project implementing a local, privacy-preserving AI system for clinical protocol question answering. The application indexes a clinical trial protocol PDF and allows users to ask natural language questions about study design, eligibility criteria, dosing schedules, safety monitoring, adverse events, and follow-up procedures.

The system uses a Retrieval-Augmented Generation (RAG) architecture with hybrid retrieval, combining semantic search with BM25 keyword search. It returns concise evidence-based answers with page-level citations, query intent detection, evidence confidence estimation, and lightweight clinical entity extraction.

This project is designed as a medical document exploration prototype and is not intended for clinical decision-making.

## Indexed Document

The current prototype was developed and tested using the C4591001 clinical trial protocol related to the BNT162 RNA-based COVID-19 vaccine study.

The protocol PDF is not included in this repository due to file size and licensing considerations. Users should download the public protocol document and place it manually at:

data/raw/C4591001_protocol.pdf

Protocol source:

https://www.nejm.org/doi/suppl/10.1056/NEJMoa2034577/suppl_file/nejmoa2034577_protocol.pdf

## Key Features

Hybrid RAG Retrieval: Combines ChromaDB semantic search with BM25 lexical search and rule-based reranking to retrieve both meaning-based passages and exact clinical/protocol terms.

Evidence-Based Answers: Generates concise answers from retrieved protocol evidence instead of relying on unsupported generation.

Page-Level Citations: Returns source document names, protocol pages, and retrieval scores for transparency.

AI / NLP Analysis Layer: Detects query intent, estimates evidence confidence, and extracts clinical/protocol entities from retrieved evidence.

Clinical Entity Extraction: Identifies protocol terms such as vaccine names, diseases, visits, dosing windows, adverse events, safety terms, and data collection procedures.

No-Answer Handling: Detects unsupported or out-of-scope questions and avoids generating answers when sufficient evidence is not found.

Audit Logging: Stores user role, question, answer, sources, timestamp, and status locally using SQLite.

FastAPI Backend: Provides API endpoints for health checks and protocol question answering.

Streamlit Frontend: Offers a professional local web interface for asking questions, viewing answers, inspecting AI/NLP analysis, and reviewing sources.

Evaluation Suite: Includes automated evaluation questions for supported answers and unsupported-question behavior.

API Testing: Includes backend API tests for health checks, valid question answering, and no-answer behavior.

## AI / NLP Components

The project includes several lightweight AI/NLP components that run locally without cloud APIs or heavy external LLMs:

Semantic embeddings using sentence-transformers/all-MiniLM-L6-v2

ChromaDB vector search for meaning-based retrieval

BM25 lexical retrieval for exact keyword, code, and protocol term matching

Rule-based reranking for clinically relevant protocol sections

Query intent classification, including eligibility, dosing, safety, study design, outcomes/data, and unsupported financial/administrative queries

Evidence confidence estimation using retrieval score and query-evidence overlap

Lightweight clinical entity extraction from retrieved protocol passages

## Indexed Document

The current indexed document is:

C4591001_protocol.pdf

This is a clinical trial protocol related to the BNT162 RNA-based COVID-19 vaccine study.

## Technology Stack

Backend: FastAPI, Python, Uvicorn

Frontend: Streamlit

Retrieval: ChromaDB, BM25, LangChain

Embeddings: Sentence Transformers

Database: SQLite

Evaluation: Python scripts with expected keyword matching

Local Storage: PDF files, vector store, audit logs, and evaluation results remain on the local machine

## Project Structure

app/api/main.py - FastAPI backend

frontend/streamlit_app.py - Streamlit frontend

src/ingest.py - PDF loading, chunking, and vector store creation

src/query.py - Hybrid retrieval using ChromaDB, BM25, and reranking

src/ask.py - CLI question answering with sources and AI/NLP analysis

src/entity_extractor.py - Query intent detection, confidence estimation, and entity extraction

src/audit_logger.py - Local SQLite audit logging

evaluation/run_evaluation.py - Automated evaluation script

evaluation/api_test.py - FastAPI backend test script

data/raw/ - Source PDF files

data/vector_store/ - ChromaDB vector database

data/audit_logs.db - Local audit log database

## Local Installation & Setup

Clone the repository:

git clone https://github.com/YOUR_USERNAME/clinical-trial-matching-engine.git

cd clinical-trial-matching-engine

Create and activate a virtual environment:

python3 -m venv venv

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Place the protocol PDF in:

data/raw/C4591001_protocol.pdf

Run ingestion:

python src/ingest.py

Start the FastAPI backend:

uvicorn app.api.main:app --reload

Start the Streamlit frontend in a second terminal:

streamlit run frontend/streamlit_app.py

Open the local application at:

http://localhost:8501

## Usage Examples

Ask from the CLI:

python src/ask.py "What is the dosing schedule?"

Run retrieval debugging:

python src/query.py "What is the study design?" --k 5

Run evaluation:

python evaluation/run_evaluation.py

Run API tests while the backend is active:

python evaluation/api_test.py

## Current Evaluation Results

The evaluation suite currently tests:

Pregnancy exclusion criteria

Inclusion criteria

Study design

Dosing schedule

Unsupported question / no-answer behavior

Current result:

Total questions: 5

Passed questions: 5

Failed questions: 0

Overall pass rate: 1.00

## Privacy

The current version runs locally. The indexed PDF, embeddings, ChromaDB vector store, generated answers, AI/NLP analysis results, and audit logs remain on the local machine.

No cloud API is required for the current implementation.

## Limitations

The answer generator is extractive and rule-based.

The clinical entity extraction layer is rule-based, not a pretrained biomedical NER model.

The current implementation is designed for one indexed clinical protocol.

The system is not intended for medical diagnosis, treatment decisions, patient-specific recommendations, or regulatory use.

No external LLM is used in the current version.

## Future Work

Optional local LLM integration

Pretrained biomedical NER model integration

Support for multiple clinical protocols

Document upload from the frontend

Docker deployment

User authentication and role-based access

Larger evaluation dataset

Comparison between semantic-only retrieval and hybrid retrieval

## Security & Medical Legal Disclaimer

Disclaimer: This software is designed strictly as a personal portfolio project and local prototype. It is not intended to replace professional medical advice, clinical diagnosis, clinical trial review, regulatory evaluation, or human expert judgment. All automated outputs should be verified by qualified professionals before any real-world use.# Clinical Protocol RAG Assistant

Local Application: http://localhost:8501

A personal portfolio project implementing a local, privacy-preserving AI system for clinical protocol question answering. The application indexes a clinical trial protocol PDF and allows users to ask natural language questions about study design, eligibility criteria, dosing schedules, safety monitoring, adverse events, and follow-up procedures.

The system uses a Retrieval-Augmented Generation (RAG) architecture with hybrid retrieval, combining semantic search with BM25 keyword search. It returns concise evidence-based answers with page-level citations, query intent detection, evidence confidence estimation, and lightweight clinical entity extraction.

This project is designed as a medical document exploration prototype and is not intended for clinical decision-making.

## Key Features

Hybrid RAG Retrieval: Combines ChromaDB semantic search with BM25 lexical search and rule-based reranking to retrieve both meaning-based passages and exact clinical/protocol terms.

Evidence-Based Answers: Generates concise answers from retrieved protocol evidence instead of relying on unsupported generation.

Page-Level Citations: Returns source document names, protocol pages, and retrieval scores for transparency.

AI / NLP Analysis Layer: Detects query intent, estimates evidence confidence, and extracts clinical/protocol entities from retrieved evidence.

Clinical Entity Extraction: Identifies protocol terms such as vaccine names, diseases, visits, dosing windows, adverse events, safety terms, and data collection procedures.

No-Answer Handling: Detects unsupported or out-of-scope questions and avoids generating answers when sufficient evidence is not found.

Audit Logging: Stores user role, question, answer, sources, timestamp, and status locally using SQLite.

FastAPI Backend: Provides API endpoints for health checks and protocol question answering.

Streamlit Frontend: Offers a professional local web interface for asking questions, viewing answers, inspecting AI/NLP analysis, and reviewing sources.

Evaluation Suite: Includes automated evaluation questions for supported answers and unsupported-question behavior.

API Testing: Includes backend API tests for health checks, valid question answering, and no-answer behavior.

## AI / NLP Components

The project includes several lightweight AI/NLP components that run locally without cloud APIs or heavy external LLMs:

Semantic embeddings using sentence-transformers/all-MiniLM-L6-v2

ChromaDB vector search for meaning-based retrieval

BM25 lexical retrieval for exact keyword, code, and protocol term matching

Rule-based reranking for clinically relevant protocol sections

Query intent classification, including eligibility, dosing, safety, study design, outcomes/data, and unsupported financial/administrative queries

Evidence confidence estimation using retrieval score and query-evidence overlap

Lightweight clinical entity extraction from retrieved protocol passages

## Indexed Document

The current indexed document is:

C4591001_protocol.pdf

This is a clinical trial protocol related to the BNT162 RNA-based COVID-19 vaccine study.

## Technology Stack

Backend: FastAPI, Python, Uvicorn

Frontend: Streamlit

Retrieval: ChromaDB, BM25, LangChain

Embeddings: Sentence Transformers

Database: SQLite

Evaluation: Python scripts with expected keyword matching

Local Storage: PDF files, vector store, audit logs, and evaluation results remain on the local machine

## Project Structure

app/api/main.py - FastAPI backend

frontend/streamlit_app.py - Streamlit frontend

src/ingest.py - PDF loading, chunking, and vector store creation

src/query.py - Hybrid retrieval using ChromaDB, BM25, and reranking

src/ask.py - CLI question answering with sources and AI/NLP analysis

src/entity_extractor.py - Query intent detection, confidence estimation, and entity extraction

src/audit_logger.py - Local SQLite audit logging

evaluation/run_evaluation.py - Automated evaluation script

evaluation/api_test.py - FastAPI backend test script

data/raw/ - Source PDF files

data/vector_store/ - ChromaDB vector database

data/audit_logs.db - Local audit log database

## Local Installation & Setup

Clone the repository:

git clone https://github.com/YOUR_USERNAME/clinical-trial-matching-engine.git

cd clinical-trial-matching-engine

Create and activate a virtual environment:

python3 -m venv venv

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Place the protocol PDF in:

data/raw/C4591001_protocol.pdf

Run ingestion:

python src/ingest.py

Start the FastAPI backend:

uvicorn app.api.main:app --reload

Start the Streamlit frontend in a second terminal:

streamlit run frontend/streamlit_app.py

Open the local application at:

http://localhost:8501

## Usage Examples

Ask from the CLI:

python src/ask.py "What is the dosing schedule?"

Run retrieval debugging:

python src/query.py "What is the study design?" --k 5

Run evaluation:

python evaluation/run_evaluation.py

Run API tests while the backend is active:

python evaluation/api_test.py

## Current Evaluation Results

The evaluation suite currently tests:

Pregnancy exclusion criteria

Inclusion criteria

Study design

Dosing schedule

Unsupported question / no-answer behavior

Current result:

Total questions: 5

Passed questions: 5

Failed questions: 0

Overall pass rate: 1.00

## Privacy

The current version runs locally. The indexed PDF, embeddings, ChromaDB vector store, generated answers, AI/NLP analysis results, and audit logs remain on the local machine.

No cloud API is required for the current implementation.

## Limitations

The answer generator is extractive and rule-based.

The clinical entity extraction layer is rule-based, not a pretrained biomedical NER model.

The current implementation is designed for one indexed clinical protocol.

The system is not intended for medical diagnosis, treatment decisions, patient-specific recommendations, or regulatory use.

No external LLM is used in the current version.

## Future Work

Optional local LLM integration

Pretrained biomedical NER model integration

Support for multiple clinical protocols

Document upload from the frontend

Docker deployment

User authentication and role-based access

Larger evaluation dataset

Comparison between semantic-only retrieval and hybrid retrieval

## Security & Medical Legal Disclaimer

Disclaimer: This software is designed strictly as a personal portfolio project and local prototype. It is not intended to replace professional medical advice, clinical diagnosis, clinical trial review, regulatory evaluation, or human expert judgment. All automated outputs should be verified by qualified professionals before any real-world use.
