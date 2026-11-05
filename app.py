"""
app.py – Streamlit entry point for the AI Fake-News & Misinformation Detector.

Run:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake-News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fake-News Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📰 Analyze Article",
        "🌐 URL Verification",
        "📊 Dashboard",
        "ℹ️ About",
    ],
)

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🔍 AI-Based Fake News & Misinformation Detector")
    st.markdown(
        """
        Welcome to the **AI-Based Fake News & Misinformation Detection** platform.

        ### What can this tool do?
        | Feature | Description |
        |---|---|
        | 📰 **Article Analysis** | Paste any article text for instant classification |
        | 🌐 **URL Verification** | Verify a news URL against fact-check databases |
        | 🤖 **Multi-Model Ensemble** | Combines ML + Transformer predictions |
        | 💡 **Explainability** | SHAP / LIME explanations for every decision |
        | 📊 **Dashboard** | Live statistics and historical trends |

        > **Get started** – choose a page from the sidebar on the left.
        """
    )

elif page == "📰 Analyze Article":
    st.title("📰 Analyze Article Text")
    st.info("Paste the full text of a news article below and click **Analyze**.")

    article_text = st.text_area("Article text", height=300, placeholder="Paste article here…")

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    if analyze_btn:
        if not article_text.strip():
            st.warning("Please enter some article text before analyzing.")
        else:
            with st.spinner("Analyzing…"):
                # TODO: wire up src.ml.predictor
                st.success("Analysis complete! (placeholder – model integration pending)")
                st.metric("Prediction", "REAL", delta="85 % confidence")

elif page == "🌐 URL Verification":
    st.title("🌐 URL Verification")
    url = st.text_input("News article URL", placeholder="https://example.com/article")
    if st.button("🔗 Verify URL"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and verifying…"):
                # TODO: wire up src.verification.fact_checker
                st.info("URL verification coming soon.")

elif page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")
    st.info("Dashboard metrics will appear here once articles have been analyzed.")

elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown(
        """
        **AI-Based Fake News & Misinformation Detection**

        This project uses a multi-layered approach combining:
        - Classical ML models (TF-IDF + XGBoost / LightGBM)
        - Transformer-based models (RoBERTa, BERT)
        - External fact-check API cross-referencing
        - SHAP / LIME explainability

        Built with ❤️ using Python, Streamlit, Hugging Face Transformers, and scikit-learn.
        """
    )

