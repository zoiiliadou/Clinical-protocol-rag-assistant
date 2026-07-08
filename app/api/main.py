from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from ask import load_vector_store, retrieve_context, generate_extractive_answer
from audit_logger import log_query
from entity_extractor import analyze_question_and_evidence


app = FastAPI(
    title="Clinical Protocol RAG API",
    description="Local RAG API for clinical protocol question answering.",
    version="1.0.0"
)


db = None


class AskRequest(BaseModel):
    question: str
    role: str = "researcher"
    k: int = 5


class SourceItem(BaseModel):
    source: str
    page: Optional[int]
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    role: str
    sources: List[SourceItem]
    analysis: Dict[str, Any]
    status: str


@app.on_event("startup")
def startup_event():
    global db
    db = load_vector_store()


@app.get("/")
def root():
    return {
        "message": "Clinical Protocol RAG API is running.",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "docs": "/docs",
            "app": "/app"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "vector_store_loaded": db is not None
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not loaded."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        results = retrieve_context(
            db=db,
            query=request.question,
            final_k=request.k
        )

        answer = generate_extractive_answer(
            query=request.question,
            results=results
        )

        analysis = analyze_question_and_evidence(
            question=request.question,
            answer=answer,
            retrieved_results=results
        )

        sources = []

        for item in results[:3]:
            metadata = item.get("metadata", {})
            page = metadata.get("page", None)

            if page is not None:
                page_display = int(page) + 1
            else:
                page_display = None

            sources.append({
                "source": metadata.get("source", "unknown source"),
                "page": page_display,
                "score": round(float(item.get("final_score", 0.0)), 4)
            })

        log_query(
            question=request.question,
            answer=answer,
            sources=sources,
            user_role=request.role,
            status="success"
        )

        return {
            "question": request.question,
            "answer": answer,
            "role": request.role,
            "sources": sources,
            "analysis": analysis,
            "status": "success"
        }

    except Exception as error:
        log_query(
            question=request.question,
            answer=str(error),
            sources=[],
            user_role=request.role,
            status="error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.get("/app", response_class=HTMLResponse)
def simple_html_app():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clinical Protocol RAG Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                background: #f8fafc;
                color: #111827;
            }
            .card {
                background: white;
                border: 1px solid #dbe3ea;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 20px;
            }
            textarea {
                width: 100%;
                height: 100px;
                font-size: 15px;
                padding: 12px;
            }
            button {
                background: #111827;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                cursor: pointer;
            }
            pre {
                background: #f1f5f9;
                padding: 12px;
                border-radius: 8px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Clinical Protocol RAG Assistant</h1>
            <p>Use the Streamlit frontend for the full interface, or test the API through /docs.</p>
            <p><b>API docs:</b> <a href="/docs">/docs</a></p>
            <p><b>Health check:</b> <a href="/health">/health</a></p>
        </div>
    </body>
    </html>
    """
