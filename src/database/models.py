"""
models.py
=========
SQLAlchemy ORM models for storing analysis history and feedback.
"""

from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class AnalysisRecord(Base):
    """Stores the result of every article analysis."""

    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Input
    source_url = Column(String(2048), nullable=True)
    article_title = Column(String(512), nullable=True)
    article_text = Column(Text, nullable=False)

    # Prediction
    prediction_label = Column(String(10), nullable=False)   # "REAL" | "FAKE"
    confidence = Column(Float, nullable=False)

    # Verification
    fact_check_verdict = Column(String(32), nullable=True)  # VERIFIED_TRUE etc.
    source_credibility_score = Column(Float, nullable=True)
    source_credibility_category = Column(String(16), nullable=True)

    # Feedback
    user_feedback = Column(String(10), nullable=True)       # "correct" | "wrong"

    def __repr__(self) -> str:
        return (
            f"<AnalysisRecord id={self.id} label={self.prediction_label} "
            f"confidence={self.confidence:.2f}>"
        )


class FeedbackRecord(Base):
    """Stores explicit user corrections for model retraining."""

    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    analysis_id = Column(Integer, nullable=False)   # FK → AnalysisRecord.id
    correct_label = Column(String(10), nullable=False)
    comment = Column(Text, nullable=True)

