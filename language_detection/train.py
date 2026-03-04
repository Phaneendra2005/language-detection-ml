"""
train.py — Train the language detection model.

Pipeline:
    1. Load dataset (CSV with columns: Text, Language)
    2. Preprocess & filter rows
    3. Extract character n-gram TF-IDF features
    4. Train LinearSVC (best balance of speed vs accuracy for langid)
    5. Wrap in a Pipeline and evaluate with cross-validation
    6. Save model + vectorizer to models/

Usage:
    python train.py                       # uses default dataset path
    python train.py --dataset my_data.csv
    python train.py --dataset my_data.csv --model-dir models/
"""

import argparse
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess
from feature_extraction import build_vectorizer

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "language_model.pkl")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_and_clean_dataset(path: str) -> pd.DataFrame:
    """Load CSV, drop nulls, run preprocessing filter."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Support both column name variants
    if "Text" not in df.columns and "text" in df.columns:
        df.rename(columns={"text": "Text", "language": "Language"}, inplace=True)

    required = {"Text", "Language"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must have columns {required}. Found: {list(df.columns)}")

    initial_len = len(df)
    df = df.dropna(subset=["Text", "Language"])
    df["Text"] = df["Text"].astype(str)

    # Apply preprocessing filter
    cleaned_rows = []
    for _, row in df.iterrows():
        cleaned, valid = preprocess(row["Text"])
        if valid:
            cleaned_rows.append({"Text": cleaned, "Language": row["Language"].strip()})

    df_clean = pd.DataFrame(cleaned_rows)
    print(f"Rows: {initial_len} → {len(df_clean)} after cleaning")
    print(f"Languages: {sorted(df_clean['Language'].unique())}")
    print(f"Samples per language:\n{df_clean['Language'].value_counts().to_string()}\n")
    return df_clean


def build_pipeline() -> Pipeline:
    """
    Full sklearn Pipeline:
        TF-IDF (char 2-4 grams)  →  LinearSVC (calibrated for probabilities)

    Why LinearSVC?
    - Consistently achieves 97-100% accuracy on language identification tasks
    - Fast to train and predict
    - Works well with high-dimensional sparse TF-IDF features
    - CalibratedClassifierCV adds .predict_proba() so we can show confidence scores
    """
    vectorizer = build_vectorizer()
    classifier = CalibratedClassifierCV(
        LinearSVC(
            C=5.0,
            max_iter=3000,
            dual=True,
            class_weight="balanced",   # handles class imbalance automatically
        ),
        cv=3,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def train(dataset_path: str = DEFAULT_DATASET, model_dir: str = DEFAULT_MODEL_DIR) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "language_model.pkl")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_and_clean_dataset(dataset_path)
    X = df["Text"].values
    y = df["Language"].values

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

    # ── 3. Build & train pipeline ─────────────────────────────────────────────
    print("\nTraining model (TF-IDF char n-grams + LinearSVC) …")
    t0 = time.time()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")

    # ── 4. Evaluate on held-out test set ──────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*55}")
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*55}")
    print("\nPer-language classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── 5. Cross-validation (optional sanity check) ───────────────────────────
    print("Running 5-fold cross-validation …")
    cv_scores = cross_val_score(
        build_pipeline(), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── 6. Save model ─────────────────────────────────────────────────────────
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved → {model_path}")

    # Save label list for reference
    labels_path = os.path.join(model_dir, "languages.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(pipeline.classes_)))
    print(f"Language labels saved → {labels_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the language detection model")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to CSV dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory to save trained model (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args()

    # Auto-generate dataset if not present
    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}. Generating …")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
        from generate_dataset import generate_dataset
        generate_dataset(args.dataset)

    train(args.dataset, args.model_dir)
