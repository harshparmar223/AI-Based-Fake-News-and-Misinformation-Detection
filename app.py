"""
app.py – TruthLens AI
An end-to-end Machine Learning & NLP Platform for Fake News & Misinformation Detection.
Academic & Production Presentation Version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.database.crud import (
    create_analysis,
    create_feedback,
    list_analyses,
    list_feedback,
    update_analysis_feedback,
)
from src.database.session import get_db, init_db
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.uncertainty import UncertaintyAssessment, compute_uncertainty
from src.ml.predictor import EnsemblePredictor, PredictionResult
from src.verification.claim_extractor import extract_claims
from src.verification.fact_checker import GoogleFactChecker
from src.verification.news_fetcher import fetch_article
from src.verification.source_credibility import score_source

load_dotenv()
init_db()

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens AI | Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom Stylesheet ────────────────────────────────────────────────────
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Cached ML Services ────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor() -> Optional[EnsemblePredictor]:
    try:
        return EnsemblePredictor.load_default()
    except Exception as exc:
        st.error(f"Error loading prediction models: {exc}")
        return None


@st.cache_resource
def load_fact_checker() -> GoogleFactChecker:
    return GoogleFactChecker()


@st.cache_resource
def load_explainer() -> LIMEExplainer:
    return LIMEExplainer(num_features=12, num_samples=250)


predictor = load_predictor()
fact_checker = load_fact_checker()
lime_explainer = load_explainer()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: #0f766e;">🔍 TruthLens AI</h2>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #64748b;">Fake News & Misinformation Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Project Overview",
            "📰 Live Article Analyzer",
            "🌐 URL & Source Verification",
            "📊 Analytics Dashboard",
            "🎓 Project Defense & Evaluation",
            "👥 Team & Methodology",
        ],
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if predictor:
        st.success(f"● Ensemble Online ({len(predictor.model_names)} Models)")
        with st.expander("Active Estimators", expanded=False):
            st.write("• **Calibrated Linear SVM** (Platt)")
            st.write("• **Random Forest** (100 Trees)")
            st.write("• **Logistic Regression** (L2)")
    else:
        st.error("● Models Offline")

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 25px; line-height: 1.4;">
            <b>Project:</b> AI-Based Misinformation Detection<br>
            <b>Stack:</b> Python 3.13 • Scikit-Learn • Streamlit • LIME • SQLite<br>
            <b>Version:</b> 1.0 (Production Release)
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 1. Page: Project Overview ─────────────────────────────────────────────────
if page == "🏠 Project Overview":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">TruthLens AI — Misinformation Detection System</h1>
            <p class="hero-subtitle">
                An end-to-end Machine Learning, NLP, and Explainable AI framework engineered to classify news articles, 
                quantify decision uncertainty, audit publisher domain credibility, and provide human-interpretable explanations.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">🎯 93.71% SVM Accuracy</span>
                <span class="hero-badge">⚡ 0.9856 ROC-AUC</span>
                <span class="hero-badge">📚 6,042 Balanced Benchmark</span>
                <span class="hero-badge">💡 LIME Token XAI</span>
                <span class="hero-badge">🛡️ Multi-Factor Uncertainty</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Top Model Accuracy", "93.71%", delta="Support Vector Machine")
    with k2:
        st.metric("Ensemble ROC-AUC", "0.9856", delta="Excellent Discrimination")
    with k3:
        st.metric("Dataset Size", "6,042", delta="Balanced 50/50 Real/Fake")
    with k4:
        st.metric("Inference Latency", "< 15 ms", delta="Sublinear TF-IDF")

    st.markdown("---")
    st.subheader("System Architecture & Detection Pipeline")

    st.markdown(
        """
        ```mermaid
        flowchart LR
            Input["Raw Input\n(Text / URL)"] --> Pre["NLP Preprocessing\n(Clean + Lemmatize)"]
            Pre --> Feat["Feature Extraction\n(TF-IDF + 8 Linguistic)"]
            Feat --> Ensemble["Ensemble Predictor\n(SVM + RF + Logistic)"]
            Ensemble --> Verdict["Probabilistic Verdict\n(REAL vs FAKE)"]
            Ensemble --> LIME["LIME Explainability\n(Token Influence)"]
            Input --> Verify["Fact-Check & Domain\n(Google API + DB)"]
            Ensemble --> Unc["Uncertainty Engine\n(Margin + Brevity)"]
            Verify --> Unc
            Verdict --> DB[("SQLite Storage\n(History + Feedback)")]
            LIME --> UI["Interactive Dashboard"]
            Unc --> UI
            DB --> UI
        ```
        """
    )

    st.markdown("### Key Technical Innovations")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="defense-card">
                <div class="defense-label">Core Machine Learning</div>
                <h4>Calibrated Ensemble</h4>
                Combines Linear Support Vector Classifier with Platt probability calibration, 
                Random Forest for non-linear interactions, and regularized Logistic Regression.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="defense-card">
                <div class="defense-label">Explainable AI</div>
                <h4>LIME Token Attributions</h4>
                Provides complete decision transparency by highlighting the exact words driving 
                classifications towards <b>FAKE</b> (red) or <b>REAL</b> (green).
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="defense-card">
                <div class="defense-label">Trust & Safety</div>
                <h4>Multi-Factor Uncertainty</h4>
                Safeguards against false confidence by penalizing borderline probabilities, 
                inter-model discordance, and short or insubstantial text inputs.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── 2. Page: Live Article Analyzer ────────────────────────────────────────────
elif page == "📰 Live Article Analyzer":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h2 style="margin: 0; font-weight: 800;">📰 Live Article Analyzer</h2>
            <p style="color: #64748b; margin: 4px 0 0 0;">
                Test any news article body or select a presentation sample to inspect the complete classification, 
                uncertainty, explainability, and fact-checking pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Presentation Preset Samples
    sample_fake_1 = (
        "BREAKING: Secret government bunker discovered under Denver International Airport containing "
        "alien technology, mind-control transmitters, and classified military bioweapons. Whistleblowers "
        "confirm subterranean tunnels connect directly to the Pentagon, operating without congressional "
        "oversight to manipulate atmospheric weather patterns and trigger coastal storms."
    )
    sample_fake_2 = (
        "Doctors stunned after discovering drinking boiled lemon water mixed with hot baking soda "
        "completely eradicates all forms of cancer in 48 hours. Major pharmaceutical conglomerates "
        "and federal health agencies have conspired to suppress this ancient natural miracle cure "
        "to protect multi-billion dollar chemotherapy revenues."
    )
    sample_real_1 = (
        "European finance ministers reached a consensus on new fiscal regulations late Wednesday, "
        "establishing stricter debt reduction targets for member nations while permitting flexibility "
        "for green infrastructure investments. The agreement concluded months of rigorous negotiations "
        "between northern and southern member states ahead of the upcoming European Parliament elections."
    )

    st.markdown("**Load Presentation Test Samples (1-Click Demo):**")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🛸 Sample 1: Denver Airport Conspiracy (Fake)", use_container_width=True):
            st.session_state["input_text"] = sample_fake_1
            st.session_state["input_source"] = "empirenews.net"
    with p2:
        if st.button("🍋 Sample 2: Miracle Cure Hoax (Fake)", use_container_width=True):
            st.session_state["input_text"] = sample_fake_2
            st.session_state["input_source"] = "worldnewsdailyreport.com"
    with p3:
        if st.button("🇪🇺 Sample 3: European Fiscal Accord (Real)", use_container_width=True):
            st.session_state["input_text"] = sample_real_1
            st.session_state["input_source"] = "reuters.com"

    article_text = st.text_area(
        "Article Body Text",
        value=st.session_state.get("input_text", ""),
        height=200,
        placeholder="Paste full news story or article text here…",
    )

    source_domain = st.text_input(
        "Publisher Domain or URL (Optional)",
        value=st.session_state.get("input_source", ""),
        placeholder="e.g. reuters.com, bbc.com, worldnewsdailyreport.com",
    )

    analyze_btn = st.button("🚀 Run Full Classification & Verification Pipeline", type="primary", use_container_width=True)

    if analyze_btn:
        if not article_text.strip():
            st.warning("Please enter or load article text to run the analysis.")
        elif not predictor:
            st.error("ML models are offline. Please verify models directory.")
        else:
            with st.spinner("Executing NLP preprocessing, multi-model inference, and evidence search…"):
                pred_result = predictor.predict(article_text)

                cred_score = None
                if source_domain.strip():
                    cred_score = score_source(source_domain.strip())

                claims_extracted = extract_claims(article_text)
                fact_result = fact_checker.search(claims_extracted.search_query)

                uncertainty = compute_uncertainty(
                    confidence=pred_result.confidence,
                    fake_probability=pred_result.fake_probability,
                    model_scores=pred_result.model_scores,
                    article_text=article_text,
                    fact_check_verdict=fact_result.verdict if fact_result else None,
                    source_credibility_score=cred_score.score if cred_score else None,
                )

                structured_words = lime_explainer.explain_structured(
                    article_text,
                    predictor.predict_proba,
                )

                # Persist to SQLite
                with get_db() as db:
                    rec = create_analysis(
                        db,
                        source_url=source_domain or None,
                        article_title=claims_extracted.primary_claim[:500],
                        article_text=article_text,
                        prediction_label=pred_result.label,
                        confidence=pred_result.confidence,
                        fact_check_verdict=fact_result.verdict,
                        source_credibility_score=cred_score.score if cred_score else None,
                        source_credibility_category=cred_score.category if cred_score else None,
                    )
                    st.session_state["last_analysis_id"] = rec.id

            # ── Hero Verdict Banner ──────────────────────────────────────────
            st.markdown("---")
            if pred_result.label == "FAKE":
                st.markdown(
                    f"""
                    <div class="verdict-box-fake">
                        <div class="verdict-header verdict-header-fake">⚠️ POTENTIALLY MISLEADING / FAKE NEWS</div>
                        <div style="font-size: 1.1rem; color: #7f1d1d; font-weight: 600;">
                            Confidence Score: <b>{pred_result.confidence:.1%}</b> • Probability of Misinformation: <b>{pred_result.fake_probability:.1%}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-box-real">
                        <div class="verdict-header verdict-header-real">✅ LIKELY RELIABLE / REAL NEWS</div>
                        <div style="font-size: 1.1rem; color: #14532d; font-weight: 600;">
                            Confidence Score: <b>{pred_result.confidence:.1%}</b> • Probability of Veracity: <b>{1.0 - pred_result.fake_probability:.1%}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            col_v1, col_v2, col_v3 = st.columns([1.5, 1.5, 2])

            with col_v1:
                # Gauge Meter for Confidence
                gauge_color = "#ef4444" if pred_result.label == "FAKE" else "#22c55e"
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=pred_result.confidence * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Ensemble Confidence", "font": {"size": 16, "family": "Inter"}},
                        number={"suffix": "%", "font": {"size": 28, "family": "Inter", "weight": "bold"}},
                        gauge={
                            "axis": {"range": [50, 100]},
                            "bar": {"color": gauge_color},
                            "steps": [
                                {"range": [50, 75], "color": "#f1f5f9"},
                                {"range": [75, 100], "color": "#e2e8f0"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_v2:
                st.markdown("#### 🛡️ Uncertainty Assessment")
                pill_class = f"pill-unc-{uncertainty.level.lower()}"
                st.markdown(
                    f"<span class='{pill_class}'>● {uncertainty.level} RISK (Index: {uncertainty.uncertainty_score:.2f})</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"**Guidance:** {uncertainty.recommendation}")
                if uncertainty.reasons:
                    with st.expander("Identified Risk Factors", expanded=True):
                        for r in uncertainty.reasons:
                            st.write(f"• {r}")

            with col_v3:
                st.markdown("#### 🤖 Model Votes Breakdown")
                model_df = pd.DataFrame(
                    [
                        {"Model": "Calibrated SVM", "Prob Fake": pred_result.model_scores.get("svm_model", 0)},
                        {"Model": "Random Forest", "Prob Fake": pred_result.model_scores.get("random_forest_model", 0)},
                        {"Model": "Logistic Reg.", "Prob Fake": pred_result.model_scores.get("logistic_model", 0)},
                    ]
                )
                fig_bar = px.bar(
                    model_df,
                    x="Prob Fake",
                    y="Model",
                    orientation="h",
                    color="Prob Fake",
                    color_continuous_scale=["#22c55e", "#ef4444"],
                    range_x=[0, 1],
                )
                fig_bar.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    xaxis_title="Probability of Fake",
                    yaxis_title="",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Explainability: LIME Token Attributions ──────────────────────
            st.markdown("---")
            st.subheader("💡 Explainable AI: Influential Words (LIME)")
            st.markdown(
                """
                LIME (*Local Interpretable Model-agnostic Explanations*) perturbs words within the article 
                to determine each token's contribution to the prediction. 
                <span class="chip-fake">Red words push prediction toward FAKE</span> 
                <span class="chip-real">Green words push prediction toward REAL</span>
                """,
                unsafe_allow_html=True,
            )

            if structured_words:
                chip_html = "<div style='margin: 12px 0 16px 0;'>"
                for item in structured_words:
                    c_class = "chip-fake" if item["influence"] == "FAKE" else "chip-real"
                    sign = "+" if item["weight"] > 0 else ""
                    chip_html += f"<span class='{c_class}'>{item['word']} ({sign}{item['weight']:.3f})</span>"
                chip_html += "</div>"
                st.markdown(chip_html, unsafe_allow_html=True)

                lime_df = pd.DataFrame(structured_words)
                fig_lime = px.bar(
                    lime_df,
                    x="weight",
                    y="word",
                    orientation="h",
                    color="influence",
                    color_discrete_map={"FAKE": "#ef4444", "REAL": "#22c55e"},
                    title="Feature Attribution Weights",
                )
                fig_lime.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=35, b=10),
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Influence on Fake Probability (Positive = Fake, Negative = Real)",
                    yaxis_title="",
                )
                st.plotly_chart(fig_lime, use_container_width=True)

            # ── Source & External Fact-Check ─────────────────────────────────
            st.markdown("---")
            c_left, c_right = st.columns(2)

            with c_left:
                st.subheader("🌐 Source Credibility Analysis")
                if cred_score:
                    st.write(f"**Evaluated Domain:** `{cred_score.domain}`")
                    st.write(f"**Credibility Tier:** `{cred_score.category}` ({cred_score.score * 100:.0f} / 100)")
                    st.info(f"**Rationale:** {cred_score.rationale}")
                else:
                    st.info("No source domain provided. Evaluation performed strictly on text semantics.")

            with c_right:
                st.subheader("🔍 External Fact-Check Cross-Referencing")
                st.write(f"**Extracted Query:** `{claims_extracted.search_query}`")
                if fact_result and fact_result.claims:
                    v_color = "red" if fact_result.verdict == "VERIFIED_FALSE" else "green"
                    st.markdown(f"**Database Status:** :{v_color}[**{fact_result.verdict}**]")
                    for c in fact_result.claims[:2]:
                        st.markdown(f"• **Claim:** *\"{c.text}\"*")
                        st.markdown(f"  **Rating:** `{c.rating}` by **{c.source}**")
                        if c.url:
                            st.markdown(f"  [Read verified fact-check]({c.url})")
                else:
                    st.write("No direct matching debunked claims found in fact-checking database.")
                    st.caption("A claim not appearing in fact-check records does not guarantee truth.")

            # ── User Feedback & Model Audit ──────────────────────────────────
            st.markdown("---")
            st.subheader("🗳️ User Feedback & Continuous Learning")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                if st.button("👍 Confirm Correct Prediction", use_container_width=True):
                    with get_db() as db:
                        update_analysis_feedback(db, st.session_state["last_analysis_id"], "correct")
                    st.success("Feedback recorded! Logged into SQLite database.")

            with f_col2:
                if st.button("👎 Flag Prediction for Retraining", use_container_width=True):
                    st.session_state["show_feedback_box"] = True

            if st.session_state.get("show_feedback_box"):
                with st.form("feedback_form"):
                    st.write("Submit Correction to Active Retraining Buffer:")
                    corr_label = st.selectbox("Correct Label", ["REAL", "FAKE"])
                    comment = st.text_input("Reason / Comment", placeholder="Why was the model incorrect?")
                    submit_corr = st.form_submit_button("Submit Correction")
                    if submit_corr:
                        with get_db() as db:
                            update_analysis_feedback(db, st.session_state["last_analysis_id"], "wrong")
                            create_feedback(
                                db,
                                analysis_id=st.session_state["last_analysis_id"],
                                correct_label=corr_label,
                                comment=comment,
                            )
                        st.success("Correction saved! Queued for future model retraining.")
                        st.session_state["show_feedback_box"] = False


# ── 3. Page: URL & Source Verification ────────────────────────────────────────
elif page == "🌐 URL & Source Verification":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h2 style="margin: 0; font-weight: 800;">🌐 News URL & Source Verification</h2>
            <p style="color: #64748b; margin: 4px 0 0 0;">
                Analyze news links in real time to assess publisher reputation and automatically scrape content.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_urls = [
        "https://www.reuters.com/business/finance/central-banks-digital-settlement",
        "https://theonion.com/study-finds-majority-of-time-spent-at-desk",
        "https://worldnewsdailyreport.com/lemon-baking-soda-miracle",
    ]
    st.write("**Try a test URL:**")
    u_c1, u_c2, u_c3 = st.columns(3)
    with u_c1:
        if st.button("📰 Reuters (High Credibility)"):
            st.session_state["url_input"] = sample_urls[0]
    with u_c2:
        if st.button("🎭 The Onion (Known Satire)"):
            st.session_state["url_input"] = sample_urls[1]
    with u_c3:
        if st.button("⚠️ World News Daily (Low Credibility)"):
            st.session_state["url_input"] = sample_urls[2]

    target_url = st.text_input("News Article URL", value=st.session_state.get("url_input", ""), placeholder="https://example.com/article...")

    if st.button("🔗 Run URL Verification", type="primary"):
        if not target_url.strip():
            st.warning("Please enter a URL to inspect.")
        else:
            with st.spinner("Inspecting domain reputation and scraping article body…"):
                cred = score_source(target_url)
                article_data = fetch_article(target_url)

            st.markdown("---")
            st.subheader("Domain Credibility Scorecard")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Domain", cred.domain)
            with sc2:
                st.metric("Credibility Score", f"{cred.score * 100:.0f} / 100")
            with sc3:
                badge_map = {"HIGH": "🟢 High Credibility", "MEDIUM": "🟡 Medium Credibility", "LOW": "🔴 Low Credibility", "SATIRE": "🎭 Known Satire"}
                st.metric("Reputation Tier", badge_map.get(cred.category, cred.category))

            st.info(f"**Reputation Analysis:** {cred.rationale}")

            st.markdown("---")
            if article_data.success:
                st.subheader(f"Extracted Article: {article_data.title or cred.domain}")
                if article_data.publish_date:
                    st.caption(f"Published Date: {article_data.publish_date}")
                with st.expander("View Scraped Article Body", expanded=False):
                    st.write(article_data.text)

                if predictor and article_data.text:
                    with st.spinner("Classifying scraped content…"):
                        url_pred = predictor.predict(article_data.text)
                    if url_pred.label == "FAKE":
                        st.error(f"Prediction: **POTENTIALLY MISLEADING** ({url_pred.confidence:.1%} confidence)")
                    else:
                        st.success(f"Prediction: **LIKELY RELIABLE** ({url_pred.confidence:.1%} confidence)")
            else:
                st.warning(f"Could not automatically parse the full body text: {article_data.error or 'Protected webpage'}")
                st.info("You can copy the text directly into the 'Live Article Analyzer' tab.")


# ── 4. Page: Analytics Dashboard ──────────────────────────────────────────────
elif page == "📊 Analytics Dashboard":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h2 style="margin: 0; font-weight: 800;">📊 Analytics & Historical Audit Dashboard</h2>
            <p style="color: #64748b; margin: 4px 0 0 0;">
                Live metrics, prediction distributions, and feedback logs stored in the SQLite database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with get_db() as db:
        records = list_analyses(db, limit=300)
        feedback_records = list_feedback(db, limit=300)

    if not records:
        st.info("No records logged yet. Run analyses in the Live Article Analyzer to populate the dashboard.")
    else:
        df = pd.DataFrame(
            [
                {
                    "ID": r.id,
                    "Date": r.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Headline / Claim": (r.article_title or r.article_text[:60]) + "…",
                    "Label": r.prediction_label,
                    "Confidence": r.confidence,
                    "Source": r.source_url or "Direct Input",
                    "Fact Check": r.fact_check_verdict or "UNVERIFIED",
                    "Feedback": r.user_feedback or "Pending",
                }
                for r in records
            ]
        )

        total_analyses = len(df)
        fake_total = (df["Label"] == "FAKE").sum()
        real_total = (df["Label"] == "REAL").sum()
        avg_conf = df["Confidence"].mean()
        confirmed_correct = (df["Feedback"] == "correct").sum()
        user_corrections = len(feedback_records)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Analyses", total_analyses)
        with m2:
            st.metric("Misinformation Flagged", f"{fake_total} ({fake_total / total_analyses:.1%})")
        with m3:
            st.metric("Reliable Content", f"{real_total} ({real_total / total_analyses:.1%})")
        with m4:
            st.metric("Average Confidence", f"{avg_conf:.1%}")

        st.markdown("---")
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_pie = px.pie(
                df,
                names="Label",
                title="Class Distribution (Historical Analyses)",
                color="Label",
                color_discrete_map={"FAKE": "#ef4444", "REAL": "#22c55e"},
                hole=0.45,
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            fig_hist = px.histogram(
                df,
                x="Confidence",
                nbins=20,
                color="Label",
                color_discrete_map={"FAKE": "#ef4444", "REAL": "#22c55e"},
                title="Confidence Score Distribution",
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("Historical Analysis Records")
        filter_col, search_col = st.columns([1, 3])
        with filter_col:
            selected_filter = st.selectbox("Filter Class", ["All", "FAKE", "REAL"])
        with search_col:
            search_query = st.text_input("Search Headline / Claim", placeholder="Filter records by keyword…")

        filtered_records = df.copy()
        if selected_filter != "All":
            filtered_records = filtered_records[filtered_records["Label"] == selected_filter]
        if search_query.strip():
            filtered_records = filtered_records[
                filtered_records["Headline / Claim"].str.contains(search_query, case=False, na=False)
            ]

        st.dataframe(filtered_records, use_container_width=True)

        if feedback_records:
            st.markdown("---")
            st.subheader("User Corrections & Retraining Queue")
            fb_df = pd.DataFrame(
                [
                    {
                        "Feedback ID": fb.id,
                        "Analysis ID": fb.analysis_id,
                        "Submitted At": fb.created_at.strftime("%Y-%m-%d %H:%M"),
                        "Correct Label": fb.correct_label,
                        "Comment": fb.comment or "No comment",
                    }
                    for fb in feedback_records
                ]
            )
            st.dataframe(fb_df, use_container_width=True)


# ── 5. Page: Project Defense & Evaluation ─────────────────────────────────────
elif page == "🎓 Project Defense & Evaluation":
    st.markdown(
        """
        <div class="hero-banner" style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 60%, #4338ca 100%);">
            <h1 class="hero-title">Academic Project Defense & Model Evaluation</h1>
            <p class="hero-subtitle">
                Comprehensive comparative analysis, confusion matrices, and empirical validation 
                prepared for academic evaluation and faculty review.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Curated ISOT & McIntire Datasets</span>
                <span class="hero-badge">37/37 Unit & Integration Tests Passed</span>
                <span class="hero-badge">Platt Probability Calibration</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Experimental Model Comparison")
    st.write(
        "Three distinct machine learning paradigms were trained on identical stratified training data "
        "(4,833 articles) and evaluated on an unseen held-out test partition (1,209 articles)."
    )

    eval_data = {
        "Classifier": ["Support Vector Machine (LinearSVC)", "Logistic Regression", "Random Forest Classifier"],
        "Accuracy": [0.9371, 0.9264, 0.8933],
        "Precision": [0.9337, 0.9118, 0.8756],
        "Recall": [0.9428, 0.9461, 0.9199],
        "F1-Score": [0.9382, 0.9286, 0.8972],
        "ROC-AUC": [0.9856, 0.9802, 0.9570],
        "Inference Speed": ["< 1 ms", "< 1 ms", "12 ms"],
    }
    eval_df = pd.DataFrame(eval_data)

    # Format percentages
    styled_eval = eval_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        styled_eval[col] = styled_eval[col].apply(lambda x: f"{x * 100:.2f}%")
    st.table(styled_eval.set_index("Classifier"))

    st.markdown("---")
    st.subheader("2. Comparative Performance Metrics Chart")
    fig_comp = px.bar(
        eval_df,
        x="Classifier",
        y=["Accuracy", "F1-Score", "ROC-AUC"],
        barmode="group",
        title="Model Metric Comparisons across Architecture Families",
        color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"],
    )
    fig_comp.update_layout(yaxis=dict(range=[0.8, 1.0]), height=350)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Test Partition Confusion Matrices")
    cm_c1, cm_c2, cm_c3 = st.columns(3)

    with cm_c1:
        st.markdown("**Calibrated SVM (Top Performer)**")
        cm_svm = np.array([[556, 41], [35, 577]])
        fig_cm1 = px.imshow(
            cm_svm,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=["Real", "Fake"],
            y=["Real", "Fake"],
            color_continuous_scale="Blues",
        )
        fig_cm1.update_layout(height=280, margin=dict(l=10, r=10, t=25, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_cm1, use_container_width=True)

    with cm_c2:
        st.markdown("**Logistic Regression**")
        cm_lr = np.array([[541, 56], [33, 579]])
        fig_cm2 = px.imshow(
            cm_lr,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=["Real", "Fake"],
            y=["Real", "Fake"],
            color_continuous_scale="Greens",
        )
        fig_cm2.update_layout(height=280, margin=dict(l=10, r=10, t=25, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_cm2, use_container_width=True)

    with cm_c3:
        st.markdown("**Random Forest**")
        cm_rf = np.array([[517, 80], [49, 563]])
        fig_cm3 = px.imshow(
            cm_rf,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=["Real", "Fake"],
            y=["Real", "Fake"],
            color_continuous_scale="Purples",
        )
        fig_cm3.update_layout(height=280, margin=dict(l=10, r=10, t=25, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_cm3, use_container_width=True)

    st.markdown("---")
    st.subheader("4. Technical Defense & Research Questions")
    with st.expander("Q1: Why was Calibrated LinearSVC superior to Random Forest?", expanded=True):
        st.write(
            "In high-dimensional sparse text classification (10,000 TF-IDF features), text representations "
            "are linearly separable across hyperplane boundaries. Linear Support Vector Machines maximize the margin "
            "between classes without overfitting. We applied 3-fold Platt scaling via `CalibratedClassifierCV` to "
            "transform raw margins into calibrated probabilities required for confidence estimation and LIME."
        )

    with st.expander("Q2: How does TruthLens prevent false alarms on brief or satirical inputs?", expanded=True):
        st.write(
            "TruthLens utilizes a multi-layered defense: (1) Domain reputation heuristics immediately detect satire domains "
            "such as The Onion or ClickHole; (2) An Uncertainty Indicator flags articles containing fewer than 30-60 words, "
            "attaching an explicit 'Data Insufficiency' warning; (3) External fact-checks from Google Fact Check Tools API "
            "cross-reference known verified debunkings."
        )


# ── 6. Page: Team & Methodology ───────────────────────────────────────────────
elif page == "👥 Team & Methodology":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <h2 style="margin: 0; font-weight: 800;">👥 Project Team & Methodology</h2>
            <p style="color: #64748b; margin: 4px 0 0 0;">
                Academic project structure, member responsibilities, and architectural specifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            """
            <div class="pro-card">
                <h4>Member 1 — Dataset & NLP</h4>
                <p style="font-size: 0.85rem; color: #64748b;">Data Engineering Lead</p>
                <ul>
                    <li>ISOT & McIntire dataset collection</li>
                    <li>HTML stripping & text cleaning</li>
                    <li>NLTK tokenisation & WordNet lemmatisation</li>
                    <li>EDA & Preprocessing notebooks</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="pro-card">
                <h4>Member 2 — Machine Learning</h4>
                <p style="font-size: 0.85rem; color: #64748b;">Algorithm & Modeling Lead</p>
                <ul>
                    <li>Sublinear TF-IDF vectorisation</li>
                    <li>Calibrated Linear SVM modeling</li>
                    <li>Random Forest & Logistic baselines</li>
                    <li>Comparative evaluation & metrics</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """
            <div class="pro-card">
                <h4>Member 3 — Verification</h4>
                <p style="font-size: 0.85rem; color: #64748b;">Fact-Checking & Scraping</p>
                <ul>
                    <li>Domain reputation heuristic scoring</li>
                    <li>Claim & keyword extraction logic</li>
                    <li>Google Fact Check API client</li>
                    <li>URL parsing & BeautifulSoup scraper</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    t4, t5 = st.columns(2)
    with t4:
        st.markdown(
            """
            <div class="pro-card">
                <h4>Member 4 — Frontend & Database</h4>
                <p style="font-size: 0.85rem; color: #64748b;">Full-Stack & Systems</p>
                <ul>
                    <li>Streamlit multi-page interface design</li>
                    <li>SQLite & SQLAlchemy ORM database</li>
                    <li>Live analytics & audit dashboard</li>
                    <li>User feedback & correction recording</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t5:
        st.markdown(
            """
            <div class="pro-card">
                <h4>Member 5 — Explainability & Integration</h4>
                <p style="font-size: 0.85rem; color: #64748b;">Lead Architecture & XAI</p>
                <ul>
                    <li>LIME local surrogate token attribution</li>
                    <li>Multi-factor uncertainty quantification</li>
                    <li>End-to-end ensemble integration</li>
                    <li>37-test suite & project defense docs</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Academic Project Specifications")
    st.markdown(
        """
        - **Primary Language:** Python 3.13
        - **Web Framework:** Streamlit 1.58
        - **Core NLP Libraries:** NLTK 3.10 (WordNet, Stopwords, Punkt)
        - **Machine Learning:** Scikit-Learn 1.9 (LinearSVC, CalibratedClassifierCV, RandomForestClassifier, LogisticRegression)
        - **Explainability:** LIME 0.2.0 (Local Interpretable Model-agnostic Explanations)
        - **Data Visualizations:** Plotly 6.8 & Matplotlib 3.11
        - **Database:** SQLite with SQLAlchemy ORM
        - **Test Framework:** Pytest 8.3 (37 unit & integration tests)
        """
    )
