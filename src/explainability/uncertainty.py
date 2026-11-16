"""
uncertainty.py
==============
Quantifies prediction uncertainty and data insufficiency for news articles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UncertaintyAssessment:
    level: str                  # "LOW", "MODERATE", "HIGH"
    uncertainty_score: float    # 0.0 (very certain) to 1.0 (highly uncertain)
    is_insufficient: bool       # True if article lacks sufficient content/substance
    reasons: List[str] = field(default_factory=list)
    recommendation: str = ""


def compute_uncertainty(
    confidence: float,
    fake_probability: float,
    model_scores: Dict[str, float],
    article_text: str,
    fact_check_verdict: Optional[str] = None,
    source_credibility_score: Optional[float] = None,
) -> UncertaintyAssessment:
    """
    Evaluate multiple uncertainty dimensions:
    1. Margin from decision boundary (distance from 0.5).
    2. Variance/disagreement among ensemble models.
    3. Text length & linguistic adequacy (insufficiency flag).
    4. Discordance between ML prediction and external fact verification.

    Parameters
    ----------
    confidence:
        Confidence in the predicted class (0.5 to 1.0).
    fake_probability:
        Raw ensemble probability of FAKE (0.0 to 1.0).
    model_scores:
        Dictionary mapping model names to individual probabilities.
    article_text:
        Raw input text.
    fact_check_verdict:
        Optional verdict from external fact checker (VERIFIED_TRUE, VERIFIED_FALSE, UNVERIFIED).
    source_credibility_score:
        Optional domain credibility score (0.0 to 1.0).

    Returns
    -------
    UncertaintyAssessment
    """
    reasons: List[str] = []
    penalty = 0.0
    words = article_text.split()
    word_count = len(words)

    # 1. Text Insufficiency Check
    is_insufficient = False
    if word_count < 30:
        is_insufficient = True
        penalty += 0.35
        reasons.append(f"Short text ({word_count} words): Limited linguistic signals for reliable NLP classification.")
    elif word_count < 60:
        penalty += 0.15
        reasons.append(f"Brief article ({word_count} words): Consider providing full article body.")

    # 2. Decision boundary margin
    # Distance from 0.5: 0.0 when p=0.5 (max uncertainty), 0.5 when p=0.0 or 1.0 (max certainty)
    margin = abs(fake_probability - 0.5)
    boundary_uncertainty = max(0.0, 1.0 - (margin * 2.0))  # 1.0 at 0.5, 0.0 at 0.0 or 1.0
    penalty += boundary_uncertainty * 0.45

    if boundary_uncertainty > 0.6:
        reasons.append(f"Ensemble probability ({fake_probability:.1%}) is close to the 50% decision threshold.")

    # 3. Model Disagreement
    if model_scores and len(model_scores) > 1:
        probs = list(model_scores.values())
        disagreement_spread = max(probs) - min(probs)
        if disagreement_spread >= 0.35:
            penalty += 0.25
            opposing = [f"{m}: {p:.0%}" for m, p in model_scores.items()]
            reasons.append(f"Model disagreement detected (spread: {disagreement_spread:.0%}): {', '.join(opposing)}")
        elif disagreement_spread >= 0.2:
            penalty += 0.10
            reasons.append("Mild divergence among ensemble models.")

    # 4. External Verification Discordance
    if fact_check_verdict:
        if fact_check_verdict == "VERIFIED_TRUE" and fake_probability > 0.6:
            penalty += 0.30
            reasons.append("Conflict: Independent fact-checkers rated this claim TRUE, but ML model flagged FAKE patterns.")
        elif fact_check_verdict == "VERIFIED_FALSE" and fake_probability < 0.4:
            penalty += 0.30
            reasons.append("Conflict: Independent fact-checkers debunked this claim, but ML model flagged REAL patterns.")

    # Final score clamped to [0.0, 1.0]
    total_score = round(min(max(penalty, 0.05), 1.0), 3)

    # Categorise level
    if total_score >= 0.65 or is_insufficient:
        level = "HIGH"
        recommendation = "High uncertainty: Do not rely solely on this prediction. Verify with established reputable news outlets."
    elif total_score >= 0.35:
        level = "MODERATE"
        recommendation = "Moderate uncertainty: Model identifies relevant patterns, but additional corroboration is recommended."
    else:
        level = "LOW"
        recommendation = "Low uncertainty: Strong consensus across models and linguistic signals."

    if not reasons:
        reasons.append("Consistent signals across models with clear class separation.")

    return UncertaintyAssessment(
        level=level,
        uncertainty_score=total_score,
        is_insufficient=is_insufficient,
        reasons=reasons,
        recommendation=recommendation,
    )
