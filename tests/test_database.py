"""
test_database.py
================
Unit tests for the database ORM models and CRUD helpers (using an in-memory SQLite DB).
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database import crud


@pytest.fixture(scope="module")
def db_session():
    """Provide an in-memory SQLite session for the duration of the module."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestAnalysisCRUD:
    def test_create_analysis(self, db_session):
        record = crud.create_analysis(
            db_session,
            article_text="Some article text",
            prediction_label="FAKE",
            confidence=0.87,
        )
        db_session.commit()
        assert record.id is not None
        assert record.prediction_label == "FAKE"

    def test_get_analysis(self, db_session):
        record = crud.create_analysis(
            db_session,
            article_text="Another article",
            prediction_label="REAL",
            confidence=0.65,
        )
        db_session.commit()
        fetched = crud.get_analysis(db_session, record.id)
        assert fetched is not None
        assert fetched.prediction_label == "REAL"

    def test_list_analyses(self, db_session):
        results = crud.list_analyses(db_session)
        assert len(results) >= 1

    def test_update_feedback(self, db_session):
        record = crud.create_analysis(
            db_session,
            article_text="Feedback test article",
            prediction_label="FAKE",
            confidence=0.9,
        )
        db_session.commit()
        updated = crud.update_analysis_feedback(db_session, record.id, "correct")
        db_session.commit()
        assert updated.user_feedback == "correct"


class TestFeedbackCRUD:
    def test_create_feedback(self, db_session):
        fb = crud.create_feedback(
            db_session,
            analysis_id=1,
            correct_label="REAL",
            comment="Model was wrong here.",
        )
        db_session.commit()
        assert fb.id is not None
        assert fb.correct_label == "REAL"

