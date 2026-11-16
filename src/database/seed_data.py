"""
seed_data.py
============
Seeds the database with representative historical analysis and feedback records
so the Analytics Dashboard and audit logs display real-world data during presentations.
"""

from __future__ import annotations

import datetime
from src.database.session import get_db, init_db
from src.database.crud import create_analysis, create_feedback, list_analyses

SAMPLE_RECORDS = [
    {
        "source_url": "https://www.reuters.com/business/finance/central-banks-digital-settlement",
        "article_title": "Central banks coordinate on cross-border digital settlement protocols",
        "article_text": "Central banking authorities announced a coordinated framework for cross-border wholesale central bank digital currency transactions following successful sandbox trials.",
        "prediction_label": "REAL",
        "confidence": 0.942,
        "fact_check_verdict": "VERIFIED_TRUE",
        "source_credibility_score": 0.95,
        "source_credibility_category": "HIGH",
        "user_feedback": "correct",
        "days_ago": 6,
    },
    {
        "source_url": "https://worldnewsdailyreport.com/lemon-baking-soda-miracle",
        "article_title": "Drinking boiled lemon water and baking soda cures all known cancers in 48 hours",
        "article_text": "Secret medical whistleblower exposes ancient natural cure for terminal tumors that pharmaceutical giants have suppressed for seventy years to protect chemotherapy profits.",
        "prediction_label": "FAKE",
        "confidence": 0.985,
        "fact_check_verdict": "VERIFIED_FALSE",
        "source_credibility_score": 0.05,
        "source_credibility_category": "LOW",
        "user_feedback": "correct",
        "days_ago": 5,
    },
    {
        "source_url": "https://www.bbc.com/news/science-environment-energy-transition",
        "article_title": "European Union ratifies updated renewable infrastructure deployment targets",
        "article_text": "Ministers representing member states formally approved legislation mandating 42.5 percent renewable energy shares by 2030, introducing streamlined permitting procedures.",
        "prediction_label": "REAL",
        "confidence": 0.961,
        "fact_check_verdict": "VERIFIED_TRUE",
        "source_credibility_score": 0.95,
        "source_credibility_category": "HIGH",
        "user_feedback": "correct",
        "days_ago": 5,
    },
    {
        "source_url": "https://empirenews.net/microchips-vaccine-telemetry-gates",
        "article_title": "Microscopic tracking chips found in ordinary seasonal influenza inoculations",
        "article_text": "Undercover investigative reports assert micro-transceivers disguised as preservatives transmit biometric telemetry to orbital satellite networks.",
        "prediction_label": "FAKE",
        "confidence": 0.991,
        "fact_check_verdict": "VERIFIED_FALSE",
        "source_credibility_score": 0.05,
        "source_credibility_category": "LOW",
        "user_feedback": "correct",
        "days_ago": 4,
    },
    {
        "source_url": "https://apnews.com/article/technology-semiconductor-fabrication-facility",
        "article_title": "New semiconductor fabrication facility commences commercial silicon wafer output",
        "article_text": "Leading chip manufacturers initiated production at their multi-billion dollar advanced semiconductor fabrication complex, addressing domestic automotive supply bottlenecks.",
        "prediction_label": "REAL",
        "confidence": 0.928,
        "fact_check_verdict": "UNVERIFIED",
        "source_credibility_score": 0.95,
        "source_credibility_category": "HIGH",
        "user_feedback": None,
        "days_ago": 4,
    },
    {
        "source_url": "https://nationalreport.net/denver-airport-alien-complex",
        "article_title": "Secret subterranean military facility discovered under Denver International Airport",
        "article_text": "Deep excavation teams reportedly breached classified underground chambers containing extraterrestrial technologies and weather modification devices.",
        "prediction_label": "FAKE",
        "confidence": 0.968,
        "fact_check_verdict": "VERIFIED_FALSE",
        "source_credibility_score": 0.05,
        "source_credibility_category": "LOW",
        "user_feedback": "correct",
        "days_ago": 3,
    },
    {
        "source_url": "https://theonion.com/study-finds-majority-of-time-spent-at-desk",
        "article_title": "Scientists announce humans now biologically adapted to survive entirely on ambient fluorescent light",
        "article_text": "A satirical investigation claims modern office workers no longer require caloric nutrition after evolving photosynthetic skin pigments.",
        "prediction_label": "FAKE",
        "confidence": 0.912,
        "fact_check_verdict": "UNVERIFIED",
        "source_credibility_score": 0.10,
        "source_credibility_category": "SATIRE",
        "user_feedback": "correct",
        "days_ago": 3,
    },
    {
        "source_url": "https://www.nature.com/articles/d41586-quantum-error-correction",
        "article_title": "Physicists demonstrate fault-tolerant logical qubits with neutral atom arrays",
        "article_text": "Experimental researchers achieved two-qubit gate fidelities surpassing error correction thresholds using optical tweezer arrays of neutral rubidium atoms.",
        "prediction_label": "REAL",
        "confidence": 0.974,
        "fact_check_verdict": "VERIFIED_TRUE",
        "source_credibility_score": 0.95,
        "source_credibility_category": "HIGH",
        "user_feedback": "correct",
        "days_ago": 2,
    },
    {
        "source_url": "https://unverified-blog.xyz/5g-weather-manipulation",
        "article_title": "5G frequency towers synchronized to trigger flash floods in municipal districts",
        "article_text": "Conspiracy commentators allege telecommunication towers emit directional microwave pulses designed to disrupt atmospheric barometric stability.",
        "prediction_label": "FAKE",
        "confidence": 0.955,
        "fact_check_verdict": "VERIFIED_FALSE",
        "source_credibility_score": 0.35,
        "source_credibility_category": "LOW",
        "user_feedback": "correct",
        "days_ago": 2,
    },
    {
        "source_url": "https://www.washingtonpost.com/climate-environment/coastal-restoration",
        "article_title": "Federal wetlands restoration initiative allocates funding for mangrove sanctuaries",
        "article_text": "The Department of the Interior released comprehensive conservation plans targeting five hundred thousand acres of vulnerable coastal estuaries.",
        "prediction_label": "REAL",
        "confidence": 0.936,
        "fact_check_verdict": "UNVERIFIED",
        "source_credibility_score": 0.95,
        "source_credibility_category": "HIGH",
        "user_feedback": None,
        "days_ago": 1,
    },
    {
        "source_url": "https://clickhole.com/inspirational-man-wakes-up-at-3am",
        "article_title": "Inspiring entrepreneur optimizes morning routine by sleeping directly inside espresso machine",
        "article_text": "Satirical overview exploring intense productivity optimization trends among tech startup founders.",
        "prediction_label": "FAKE",
        "confidence": 0.887,
        "fact_check_verdict": "UNVERIFIED",
        "source_credibility_score": 0.10,
        "source_credibility_category": "SATIRE",
        "user_feedback": "correct",
        "days_ago": 1,
    },
    {
        "source_url": "https://www.ft.com/content/global-maritime-decarbonisation",
        "article_title": "International Maritime Organization formally enacts green corridor shipping standards",
        "article_text": "Global maritime shipping regulators adopted mandatory carbon intensity indicator frameworks to accelerate transition to methanol and green ammonia propulsion.",
        "prediction_label": "REAL",
        "confidence": 0.958,
        "fact_check_verdict": "VERIFIED_TRUE",
        "source_credibility_score": 0.95,
        "source_credibility_category": "HIGH",
        "user_feedback": "correct",
        "days_ago": 0,
    },
    {
        "source_url": "https://mystery-daily-news.org/proxima-centauri-signals",
        "article_title": "Extraterrestrial civilization broadcasts mathematical sequence towards Earth",
        "article_text": "Unverified claims assert radio astronomers intercepted prime number sequences originating from the nearest stellar system.",
        "prediction_label": "FAKE",
        "confidence": 0.814,
        "fact_check_verdict": "UNVERIFIED",
        "source_credibility_score": 0.45,
        "source_credibility_category": "MEDIUM",
        "user_feedback": "wrong",
        "days_ago": 0,
    },
]


def seed():
    init_db()
    with get_db() as db:
        existing = list_analyses(db)
        if len(existing) >= len(SAMPLE_RECORDS):
            print(f"Database already contains {len(existing)} records. Skipping seeding.")
            return

        print(f"Seeding {len(SAMPLE_RECORDS)} realistic sample records…")
        for item in SAMPLE_RECORDS:
            days = item.pop("days_ago")
            created_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days, hours=2)
            rec = create_analysis(db, **item)
            rec.created_at = created_at
            db.flush()

            if item["user_feedback"] == "wrong":
                create_feedback(
                    db,
                    analysis_id=rec.id,
                    correct_label="REAL",
                    comment="Sensationalist phrasing led model to classify as Fake, but article actually cited genuine SETI radio observations.",
                )

        print("Database seeded successfully with rich demo history!")


if __name__ == "__main__":
    seed()
