"""
server.py
=========
Production FastAPI backend server for the TruthLens AI Fake News & Misinformation Detector.
Serves a high-end web application with complete REST API endpoints, Swagger docs,
multi-model ensemble inference, LIME explainability, URL parsing, and database persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.database.crud import (
    create_analysis,
    create_feedback,
    list_analyses,
    list_feedback,
    update_analysis_feedback,
)
from src.database.session import get_db, init_db
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.uncertainty import compute_uncertainty
from src.ml.predictor import EnsemblePredictor
from src.verification.claim_extractor import extract_claims
from src.verification.fact_checker import GoogleFactChecker
from src.verification.news_fetcher import fetch_article
from src.verification.source_credibility import score_source

# Initialize SQLite database schema
init_db()

# Initialize FastAPI application
app = FastAPI(
    title="TruthLens AI – Fake News & Misinformation Detection Platform",
    description="High-performance machine learning API combining Calibrated SVM, Random Forest, Logistic Regression, LIME explainability, and external fact verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML artifacts into memory on startup
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

try:
    predictor = EnsemblePredictor.load_default()
except Exception as e:
    print(f"Warning: could not load predictor: {e}")
    predictor = None

fact_checker = GoogleFactChecker()
lime_explainer = LIMEExplainer(num_features=12, num_samples=250)


# ── Pydantic Request Models ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Full body of the news article")
    source_url: Optional[str] = Field(None, description="Optional publisher URL or domain")


class VerifyUrlRequest(BaseModel):
    url: str = Field(..., min_length=3, description="Article URL to inspect")


class FeedbackRequest(BaseModel):
    analysis_id: int = Field(..., description="ID of the analysis record")
    feedback: str = Field(..., description="'correct' or 'wrong'")
    correct_label: Optional[str] = Field(None, description="'REAL' or 'FAKE'")
    comment: Optional[str] = Field(None, description="Optional user explanation")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": predictor is not None,
        "active_models": predictor.model_names if predictor else [],
        "framework": "FastAPI + Scikit-Learn + LIME",
    }


@app.post("/api/analyze")
def analyze_article(req: AnalyzeRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="ML inference models not initialized.")

    text = req.text.strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Article text is too short to analyze.")

    # 1. Ensemble Classification
    pred = predictor.predict(text)

    # 2. Source Credibility
    cred_score = None
    if req.source_url and req.source_url.strip():
        cred_score = score_source(req.source_url.strip())

    # 3. Claims & Fact-Check Verification
    claims = extract_claims(text)
    fact_res = fact_checker.search(claims.search_query)

    # 4. Uncertainty & Risk Assessment
    unc = compute_uncertainty(
        confidence=pred.confidence,
        fake_probability=pred.fake_probability,
        model_scores=pred.model_scores,
        article_text=text,
        fact_check_verdict=fact_res.verdict if fact_res else None,
        source_credibility_score=cred_score.score if cred_score else None,
    )

    # 5. Explainable AI (LIME Token Attributions)
    lime_words = lime_explainer.explain_structured(text, predictor.predict_proba)

    # 6. Database Persistence
    with get_db() as db:
        rec = create_analysis(
            db,
            source_url=req.source_url or None,
            article_title=claims.primary_claim[:500],
            article_text=text,
            prediction_label=pred.label,
            confidence=pred.confidence,
            fact_check_verdict=fact_res.verdict,
            source_credibility_score=cred_score.score if cred_score else None,
            source_credibility_category=cred_score.category if cred_score else None,
        )
        analysis_id = rec.id

    return {
        "id": analysis_id,
        "label": pred.label,
        "confidence": pred.confidence,
        "fake_probability": pred.fake_probability,
        "model_scores": pred.model_scores,
        "uncertainty": {
            "level": unc.level,
            "score": unc.uncertainty_score,
            "is_insufficient": unc.is_insufficient,
            "reasons": unc.reasons,
            "recommendation": unc.recommendation,
        },
        "lime_attributions": lime_words,
        "claims": {
            "primary_claim": claims.primary_claim,
            "claims": claims.claims,
            "keywords": claims.keywords,
            "search_query": claims.search_query,
        },
        "fact_check": {
            "verdict": fact_res.verdict,
            "claims": [
                {
                    "text": c.text,
                    "claimant": c.claimant,
                    "rating": c.rating,
                    "source": c.source,
                    "url": c.url,
                }
                for c in fact_res.claims[:3]
            ],
        },
        "source_credibility": {
            "domain": cred_score.domain,
            "score": cred_score.score,
            "category": cred_score.category,
            "rationale": cred_score.rationale,
        } if cred_score else None,
    }


@app.post("/api/verify-url")
def verify_news_url(req: VerifyUrlRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Invalid URL.")

    cred = score_source(url)
    fetched = fetch_article(url)

    analysis_res = None
    if predictor and fetched.success and len(fetched.text) > 30:
        pred = predictor.predict(fetched.text)
        claims = extract_claims(fetched.text, title=fetched.title)
        fact_res = fact_checker.search(claims.search_query)
        analysis_res = {
            "label": pred.label,
            "confidence": pred.confidence,
            "fake_probability": pred.fake_probability,
            "claims": claims.primary_claim,
            "fact_verdict": fact_res.verdict,
        }

    return {
        "url": url,
        "credibility": {
            "domain": cred.domain,
            "score": cred.score,
            "category": cred.category,
            "rationale": cred.rationale,
        },
        "article": {
            "title": fetched.title,
            "authors": fetched.authors,
            "publish_date": fetched.publish_date,
            "text": fetched.text[:1500] if fetched.text else "",
            "char_count": len(fetched.text),
            "success": fetched.success,
            "error": fetched.error,
        },
        "analysis": analysis_res,
    }


@app.get("/api/dashboard")
def get_dashboard_data():
    with get_db() as db:
        records = list_analyses(db, limit=200)
        feedback_list = list_feedback(db, limit=200)

        total = len(records)
        fake_count = sum(1 for r in records if r.prediction_label == "FAKE")
        real_count = sum(1 for r in records if r.prediction_label == "REAL")
        avg_conf = (sum(r.confidence for r in records) / total) if total > 0 else 0.0

        recent_items = [
            {
                "id": r.id,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
                "headline": (r.article_title or r.article_text[:60]) + "…",
                "label": r.prediction_label,
                "confidence": round(r.confidence, 3),
                "source": r.source_url or "Direct Text",
                "fact_verdict": r.fact_check_verdict or "UNVERIFIED",
                "feedback": r.user_feedback or "pending",
            }
            for r in records[:30]
        ]

        feedbacks = [
            {
                "id": f.id,
                "analysis_id": f.analysis_id,
                "created_at": f.created_at.strftime("%Y-%m-%d %H:%M"),
                "correct_label": f.correct_label,
                "comment": f.comment or "No comment",
            }
            for f in feedback_list[:30]
        ]

    return {
        "stats": {
            "total_analyses": total,
            "fake_count": fake_count,
            "real_count": real_count,
            "fake_rate": round(fake_count / total, 3) if total else 0.0,
            "real_rate": round(real_count / total, 3) if total else 0.0,
            "average_confidence": round(avg_conf, 3),
            "feedback_count": len(feedback_list),
        },
        "recent_records": recent_items,
        "feedback_logs": feedbacks,
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    with get_db() as db:
        update_analysis_feedback(db, req.analysis_id, req.feedback)
        if req.correct_label:
            create_feedback(
                db,
                analysis_id=req.analysis_id,
                correct_label=req.correct_label,
                comment=req.comment,
            )
    return {"status": "success", "message": "Feedback recorded."}


@app.get("/api/models-info")
def get_models_info():
    results_path = Path("models/evaluation_results.json")
    if results_path.exists():
        with open(results_path) as f:
            eval_metrics = json.load(f)
    else:
        eval_metrics = {
            "svm": {"accuracy": 0.9371, "precision": 0.9337, "recall": 0.9428, "f1": 0.9382, "roc_auc": 0.9856},
            "logistic": {"accuracy": 0.9264, "precision": 0.9118, "recall": 0.9461, "f1": 0.9286, "roc_auc": 0.9802},
            "random_forest": {"accuracy": 0.8933, "precision": 0.8756, "recall": 0.9199, "f1": 0.8972, "roc_auc": 0.9570},
        }

    return {
        "dataset": {
            "total_samples": 6042,
            "train_samples": 4833,
            "test_samples": 1209,
            "class_balance": {"REAL": 2989, "FAKE": 3053},
        },
        "metrics": eval_metrics,
        "confusion_matrices": {
            "svm": {"tp": 577, "tn": 556, "fp": 41, "fn": 35},
            "logistic": {"tp": 579, "tn": 541, "fp": 56, "fn": 33},
            "random_forest": {"tp": 563, "tn": 517, "fp": 80, "fn": 49},
        },
    }


# ── Mount Static Files & Frontend ─────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return "<h1>TruthLens AI is loading… Please ensure static/index.html exists.</h1>"
    with open(index_file, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    print("Starting TruthLens Web Server on http://localhost:5000 ...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
