"""
test_integration.py
===================
End-to-end integration tests verifying complete pipeline execution.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database import crud
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.uncertainty import compute_uncertainty
from src.ml.predictor import EnsemblePredictor
from src.verification.claim_extractor import extract_claims
from src.verification.fact_checker import GoogleFactChecker
from src.verification.source_credibility import score_source


def test_full_pipeline_run():
    # 1. Setup in-memory DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # 2. Input article
    raw_article = (
        "Breaking revelation claims secretly deployed 5G towers caused the recent "
        "influenza spread across major cities according to unverified blog reports."
    )
    domain = "worldnewsdailyreport.com"

    # 3. Source Credibility
    cred_score = score_source(domain)
    assert cred_score.category == "LOW"

    # 4. Model Prediction
    predictor = EnsemblePredictor.load_default()
    pred = predictor.predict(raw_article)
    assert pred.label in ("REAL", "FAKE")

    # 5. Claims and Fact Checking
    claims = extract_claims(raw_article)
    assert len(claims.keywords) > 0
    fact_checker = GoogleFactChecker()
    fact_res = fact_checker.search(claims.search_query)
    assert fact_res.verdict in ("VERIFIED_FALSE", "VERIFIED_TRUE", "UNVERIFIED")

    # 6. Uncertainty Assessment
    unc = compute_uncertainty(
        confidence=pred.confidence,
        fake_probability=pred.fake_probability,
        model_scores=pred.model_scores,
        article_text=raw_article,
        fact_check_verdict=fact_res.verdict,
        source_credibility_score=cred_score.score,
    )
    assert unc.level in ("LOW", "MODERATE", "HIGH")

    # 7. LIME Explanation
    explainer = LIMEExplainer(num_features=5, num_samples=80)
    attributions = explainer.explain_structured(raw_article, predictor.predict_proba)
    assert isinstance(attributions, list)

    # 8. Database Persistence
    record = crud.create_analysis(
        session,
        source_url=domain,
        article_title=claims.primary_claim,
        article_text=raw_article,
        prediction_label=pred.label,
        confidence=pred.confidence,
        fact_check_verdict=fact_res.verdict,
        source_credibility_score=cred_score.score,
        source_credibility_category=cred_score.category,
    )
    session.commit()
    assert record.id is not None

    # 9. User Feedback Submission
    updated = crud.update_analysis_feedback(session, record.id, "correct")
    session.commit()
    assert updated.user_feedback == "correct"

    session.close()
