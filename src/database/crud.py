"""
crud.py
=======
Create / Read / Update / Delete helpers for database models.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from src.database.models import AnalysisRecord, FeedbackRecord


# ── AnalysisRecord ────────────────────────────────────────────────────────────

def create_analysis(db: Session, **kwargs) -> AnalysisRecord:
    """Persist a new analysis result and return the created record."""
    record = AnalysisRecord(**kwargs)
    db.add(record)
    db.flush()
    return record


def get_analysis(db: Session, record_id: int) -> Optional[AnalysisRecord]:
    return db.get(AnalysisRecord, record_id)


def list_analyses(
    db: Session,
    limit: int = 100,
    offset: int = 0,
) -> List[AnalysisRecord]:
    return (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def update_analysis_feedback(
    db: Session, record_id: int, feedback: str
) -> Optional[AnalysisRecord]:
    record = get_analysis(db, record_id)
    if record:
        record.user_feedback = feedback
        db.flush()
    return record


# ── FeedbackRecord ────────────────────────────────────────────────────────────

def create_feedback(
    db: Session,
    analysis_id: int,
    correct_label: str,
    comment: Optional[str] = None,
) -> FeedbackRecord:
    record = FeedbackRecord(
        analysis_id=analysis_id,
        correct_label=correct_label,
        comment=comment,
    )
    db.add(record)
    db.flush()
    return record


def list_feedback(db: Session, limit: int = 500) -> List[FeedbackRecord]:
    return db.query(FeedbackRecord).order_by(FeedbackRecord.created_at.desc()).limit(limit).all()

