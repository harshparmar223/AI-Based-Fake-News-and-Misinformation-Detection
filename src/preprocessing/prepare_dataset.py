"""
prepare_dataset.py
==================
Organises raw datasets into True.csv / Fake.csv and executes full preprocessing
pipeline to produce cleaned train/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.preprocessing.text_cleaner import clean_text


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def organize_raw_files(df: pd.DataFrame) -> None:
    """Split dataset into True.csv and Fake.csv to match ISOT format."""
    true_df = df[df["label"].str.upper() == "REAL"].copy()
    fake_df = df[df["label"].str.upper() == "FAKE"].copy()

    true_df["subject"] = "politics"
    true_df["date"] = "2026-01-01"
    fake_df["subject"] = "news"
    fake_df["date"] = "2026-01-01"

    cols = ["title", "text", "subject", "date"]
    true_path = RAW_DIR / "True.csv"
    fake_path = RAW_DIR / "Fake.csv"

    true_df[cols].to_csv(true_path, index=False)
    fake_df[cols].to_csv(fake_path, index=False)
    logger.info(f"Saved {len(true_df)} real articles to {true_path}")
    logger.info(f"Saved {len(fake_df)} fake articles to {fake_path}")


def prepare_and_split_data(
    raw_path: Path | str = RAW_DIR / "fake_or_real_news.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean raw dataset, remove duplicates, generate features and split into train/test sets.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    logger.info(f"Loading raw dataset from {raw_path}…")
    df = pd.read_csv(raw_path)

    # Standardise column names
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Basic cleaning
    initial_count = len(df)
    df = df.dropna(subset=["text", "label"])
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.upper().str.strip()

    # Filter out empty or whitespace-only articles
    df = df[df["text"].str.len() > 30]

    # Deduplicate based on text
    df = df.drop_duplicates(subset=["text"])
    logger.info(f"Cleaned dataset from {initial_count} down to {len(df)} unique records.")

    # Generate True.csv and Fake.csv
    organize_raw_files(df)

    # Encode label: 1 = FAKE, 0 = REAL
    df["target"] = (df["label"] == "FAKE").astype(int)

    logger.info("Applying text cleaning pipeline (HTML strip, lowercasing, punctuation)…")
    df["clean_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))
    df["char_count"] = df["clean_text"].apply(len)

    # Remove articles where cleaning left almost nothing
    df = df[df["word_count"] >= 5]
    logger.info(f"Final valid articles: {len(df)}")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved training set ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved testing set ({len(test_df)} rows) to {test_path}")

    summary = {
        "total_records": len(df),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "class_distribution": df["label"].value_counts().to_dict(),
        "avg_words_real": float(df[df["target"] == 0]["word_count"].mean()),
        "avg_words_fake": float(df[df["target"] == 1]["word_count"].mean()),
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.success("Data preprocessing completed successfully.")
    return train_df, test_df


if __name__ == "__main__":
    prepare_and_split_data()
