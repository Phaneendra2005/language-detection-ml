"""
predict.py — Predict the language of a given text.
"""

import argparse
import os
import pickle
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "language_model.pkl")

CONFIDENCE_THRESHOLD = 0.30

LANGUAGE_FLAGS = {
    "English": "🇬🇧",
    "French": "🇫🇷",
    "Spanish": "🇪🇸",
    "German": "🇩🇪",
    "Italian": "🇮🇹",
    "Portuguese": "🇵🇹",
    "Russian": "🇷🇺",
    "Arabic": "🇸🇦",
    "Chinese": "🇨🇳",
    "Japanese": "🇯🇵",
    "Korean": "🇰🇷",
    "Hindi": "🇮🇳",
    "Dutch": "🇳🇱",
    "Swedish": "🇸🇪",
    "Turkish": "🇹🇷",
    "Polish": "🇵🇱",
    "Vietnamese": "🇻🇳",
    "Tamil": "🇮🇳",
    "Urdu": "🇵🇰",
    "Greek": "🇬🇷",
    "Danish": "🇩🇰",
    "Telugu": "🇮🇳",
}

_cached_model = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run python train.py first."
        )

    with open(model_path, "rb") as f:
        _cached_model = pickle.load(f)

    return _cached_model


def is_valid_text(text: str) -> bool:
    """
    Validate that input contains actual language characters.
    Reject numbers, symbols, emoji, etc.
    """
    if not text.strip():
        return False

    # Must contain at least one letter from major scripts
    pattern = r"[A-Za-z\u0C00-\u0C7F\u0600-\u06FF\u4E00-\u9FFF\u0900-\u097F]"
    return bool(re.search(pattern, text))


def predict_language(text: str, model_path: str = DEFAULT_MODEL_PATH, return_all=False):

    # Validate input first
    if not is_valid_text(text):
        return {
            "valid": False,
            "language": None,
            "confidence": None,
            "flag": "❌",
            "message": "Please enter valid text containing real words.",
            "all_scores": {},
        }

    cleaned, valid = preprocess(text)

    if not valid:
        return {
            "valid": False,
            "language": None,
            "confidence": None,
            "flag": "❌",
            "message": "Invalid text — please enter a meaningful sentence.",
            "all_scores": {},
        }

    model = load_model(model_path)

    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_

    top_idx = proba.argmax()
    language = classes[top_idx]
    confidence = float(proba[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "valid": True,
            "language": None,
            "confidence": confidence,
            "flag": "⚠️",
            "message": "Text is ambiguous. Please enter a longer sentence.",
            "all_scores": dict(zip(classes, proba.tolist())) if return_all else {},
        }

    flag = LANGUAGE_FLAGS.get(language, "🌐")

    result = {
        "valid": True,
        "language": language,
        "confidence": confidence,
        "flag": flag,
        "message": f"{flag} Detected Language: {language} ({confidence*100:.1f}% confidence)",
    }

    if return_all:
        result["all_scores"] = dict(
            sorted(zip(classes, proba.tolist()), key=lambda x: x[1], reverse=True)
        )
    else:
        result["all_scores"] = {}

    return result


def print_result(result: dict):
    print("\n", result["message"], "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str)
    args = parser.parse_args()

    if args.text:
        result = predict_language(args.text)
        print_result(result)
    else:
        while True:
            text = input("Enter text: ")
            if text.lower() in ["exit", "quit"]:
                break
            result = predict_language(text)
            print_result(result)