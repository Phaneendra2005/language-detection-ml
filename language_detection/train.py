"""
train.py — Train the language detection model
"""

import argparse
import os
import sys
import pickle
import time
import re

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


# ─────────────────────────────────────────────────────────────
# TEXT VALIDATION (same logic as prediction)
# ─────────────────────────────────────────────────────────────

def is_valid_text(text: str) -> bool:
    """
    Accept only real language characters.
    Reject numbers / symbols / emoji.
    """
    if not text.strip():
        return False

    pattern = r"[A-Za-z\u0C00-\u0C7F\u0600-\u06FF\u4E00-\u9FFF\u0900-\u097F]"
    return bool(re.search(pattern, text))


# ─────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_and_clean_dataset(path: str) -> pd.DataFrame:

    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    if "Text" not in df.columns and "text" in df.columns:
        df.rename(columns={"text": "Text", "language": "Language"}, inplace=True)

    required = {"Text", "Language"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain Text and Language columns")

    df = df.dropna(subset=["Text", "Language"])
    df["Text"] = df["Text"].astype(str)

    cleaned_rows = []

    for _, row in df.iterrows():

        text = row["Text"]

        # Reject invalid text
        if not is_valid_text(text):
            continue

        cleaned, valid = preprocess(text)

        if valid:
            cleaned_rows.append({
                "Text": cleaned,
                "Language": row["Language"].strip()
            })

    df_clean = pd.DataFrame(cleaned_rows)
    # Remove languages with too few samples
    counts = df_clean["Language"].value_counts()
    valid_langs = counts[counts >= 3].index
    df_clean = df_clean[df_clean["Language"].isin(valid_langs)]

    print(f"\nDataset cleaned")
    print(f"Samples: {len(df_clean)}")
    print(f"Languages: {sorted(df_clean['Language'].unique())}")
    print(df_clean["Language"].value_counts())

    return df_clean


# ─────────────────────────────────────────────────────────────
# VECTOR FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def build_vectorizer():

    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),        # Best for language detection
        min_df=2,
        max_features=50000,
        lowercase=True
    )


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def build_pipeline():

    vectorizer = build_vectorizer()

    classifier = CalibratedClassifierCV(
        LinearSVC(
            C=5.0,
            max_iter=3000,
            class_weight="balanced"
        ),
        cv=2
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])

    return pipeline


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train(dataset_path=DEFAULT_DATASET, model_dir=DEFAULT_MODEL_DIR):

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "language_model.pkl")

    df = load_and_clean_dataset(dataset_path)

    X = df["Text"].values
    y = df["Language"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    pipeline = build_pipeline()

    print("\nTraining model...")

    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Training completed in {elapsed:.2f}s")

    # ───────── Evaluation ─────────

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("="*60)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ───────── Cross Validation ─────────

    print("\nRunning cross-validation...")

    cv_scores = cross_val_score(
        build_pipeline(),
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1
    )

    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ───────── Save Model ─────────

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved → {model_path}")

    labels_path = os.path.join(model_dir, "languages.txt")

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(pipeline.classes_)))

    print(f"Language list saved → {labels_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR
    )

    args = parser.parse_args()

    train(args.dataset, args.model_dir)