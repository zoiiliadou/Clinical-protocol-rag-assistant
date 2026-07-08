import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict


AUDIT_DB_PATH = Path("data/audit_logs.db")


def init_audit_db() -> None:
    """
    Create the local audit log database if it does not already exist.
    """
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_role TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def log_query(
    question: str,
    answer: str,
    sources: List[Dict],
    user_role: str = "researcher",
    status: str = "success"
) -> None:
    """
    Store a local audit log entry for a question-answer interaction.
    """
    init_audit_db()

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO query_logs (
            timestamp,
            user_role,
            question,
            answer,
            sources_json,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            user_role,
            question,
            answer,
            json.dumps(sources, ensure_ascii=False),
            status
        )
    )

    conn.commit()
    conn.close()
