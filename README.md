# Clinical Protocol RAG Assistant

Local AI/RAG assistant for clinical protocol question answering with hybrid retrieval, citations, evidence confidence, and lightweight clinical NLP analysis.

This is a personal portfolio project designed as a local, privacy-preserving medical document exploration prototype. It is not intended for clinical decision-making.

## Overview

The application indexes a clinical trial protocol PDF and allows users to ask natural language questions about study design, eligibility criteria, dosing schedules, safety monitoring, adverse events, and follow-up procedures.

The system uses a Retrieval-Augmented Generation (RAG) pipeline that combines semantic retrieval, keyword retrieval, and rule-based analysis to return evidence-based answers with page-level sources.

## Key Features

Hybrid RAG Retrieval: Combines ChromaDB semantic search with BM25 keyword search and rule-based reranking.

Evidence-Based Answers: Generates concise answers from retrieved protocol evidence.

Page-Level Citations: Returns source document names, protocol pages, and retrieval scores.

AI / NLP Analysis: Detects query intent, estimates evidence confidence, and extracts clinical/protocol entities.

No-Answer Handling: Avoids answering unsupported or out-of-scope questions when sufficient evidence is not found.

Audit Logging: Stores user role, question, answer, sources, timestamp, and status locally using SQLite.

FastAPI Backend: Provides API endpoints for health checks and protocol question answering.

Streamlit Frontend: Provides a local web interface for asking questions, viewing answers, AI/NLP analysis, and sources.

Evaluation: Includes automated evaluation and API tests.

## Indexed Document

The prototype was developed and tested using the C4591001 clinical trial protocol related to the BNT162 RNA-based COVID-19 vaccine study.

The protocol PDF is not included in this repository due to file size and licensing considerations.

Download the public protocol PDF from:

https://www.nejm.org/doi/suppl/10.1056/NEJMoa2034577/suppl_file/nejmoa2034577_protocol.pdf

Then place it manually at:

data/raw/C4591001_protocol.pdf

## Technology Stack

Backend: FastAPI, Python, Uvicorn

Frontend: Streamlit

Retrieval: ChromaDB, BM25, LangChain

Embeddings: Sentence Transformers

Database: SQLite

Evaluation: Python scripts

## Project Structure

app/api/main.py - FastAPI backend

frontend/streamlit_app.py - Streamlit frontend

src/ingest.py - PDF ingestion and vector store creation

src/query.py - Hybrid retrieval

src/ask.py - CLI question answering

src/entity_extractor.py - Query intent, confidence estimation, and entity extraction

src/audit_logger.py - SQLite audit logging

evaluation/run_evaluation.py - Evaluation script

evaluation/api_test.py - API test script

## Local Installation & Setup

Clone the repository:

git clone https://github.com/zoiiliadou/clinical-protocol-rag-assistant.git

cd clinical-protocol-rag-assistant

Create and activate a virtual environment:

python3 -m venv venv

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Download the protocol PDF and place it at:

data/raw/C4591001_protocol.pdf

Run ingestion:

python src/ingest.py

Start the FastAPI backend:

uvicorn app.api.main:app --reload

Start the Streamlit frontend in a second terminal:

streamlit run frontend/streamlit_app.py

Open the application at:

http://localhost:8501

## Usage Examples

Ask from the CLI:

python src/ask.py "What is the dosing schedule?"

Run evaluation:

python evaluation/run_evaluation.py

Run API tests while the backend is running:

python evaluation/api_test.py

## Evaluation Results

Current evaluation:

Total questions: 5

Passed questions: 5

Failed questions: 0

Overall pass rate: 1.00

The evaluation includes supported protocol questions and unsupported-question / no-answer behavior.

## Privacy

The current version runs locally. The PDF, embeddings, vector database, generated answers, AI/NLP analysis results, and audit logs remain on the local machine.

No cloud API is required for the current implementation.

## Limitations

The answer generator is extractive and rule-based.

The clinical entity extraction layer is rule-based, not a pretrained biomedical NER model.

The current version is designed for one indexed clinical protocol.

No external LLM is used in the current version.

## Future Work

Optional local LLM integration

Pretrained biomedical NER integration

Support for multiple clinical protocols

Document upload from the frontend

Docker deployment

Larger evaluation dataset

## Disclaimer

This software is designed strictly as a personal portfolio project and local prototype. It does not replace professional medical advice, clinical diagnosis, clinical trial review, regulatory evaluation, or human expert judgment.
