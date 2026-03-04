"""
feature_extraction.py — Character n-gram feature extraction.

Uses scikit-learn's TfidfVectorizer with character n-grams as features.
Character n-grams are the best features for language identification because:
  - They capture orthographic patterns unique to each script/language.
  - They are robust to unknown/rare words.
  - They work across all Unicode scripts without tokenisation.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


def build_vectorizer(
    analyzer: str = "char_wb",
    ngram_range: tuple = (2, 4),
    max_features: int = 50_000,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Create a fresh TF-IDF vectorizer with character n-gram settings.

    analyzer='char_wb':
        Uses character n-grams inside word boundaries (pads words with
        spaces). This slightly outperforms plain 'char' for language ID.

    ngram_range=(2,4):
        Bi-, tri-, and quad-grams capture letter combinations that are
        highly language-specific (e.g., 'sch' in German, 'qu' in French/
        Spanish, Cyrillic pairs in Russian, CJK bigrams …).

    max_features=50_000:
        Limits memory while keeping the most discriminating n-grams.

    sublinear_tf=True:
        Applies 1 + log(tf) instead of raw tf — helps with long texts.
    """
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        strip_accents=None,   # Keep accents — they are language-discriminating
        lowercase=True,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved → {path}")


def load_vectorizer(path: str) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)
