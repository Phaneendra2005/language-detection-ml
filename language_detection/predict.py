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


def load_model(model_path=DEFAULT_MODEL_PATH):

    global _cached_model

    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")

    with open(model_path, "rb") as f:
        _cached_model = pickle.load(f)

    return _cached_model


# ─────────────────────────────────────────────
# TEXT VALIDATION
# ─────────────────────────────────────────────

def is_valid_text(text):

    text = text.strip()

    if len(text) < 3:
        return False

    # Must contain letters from supported scripts
    letter_pattern = r"[A-Za-z\u0900-\u097F\u0C00-\u0C7F\u0600-\u06FF\u4E00-\u9FFF]"
    if not re.search(letter_pattern, text):
        return False

    # Reject numbers or symbols only
    if re.fullmatch(r"[0-9\W_]+", text):
        return False

    # Reject repeated characters
    if len(set(text.lower())) <= 2:
        return False

    # Latin text must contain vowel
    if re.match(r"^[A-Za-z\s]+$", text):
        if not re.search(r"[aeiou]", text.lower()):
            return False

    return True


# ─────────────────────────────────────────────
# LANGUAGE PREDICTION
# ─────────────────────────────────────────────

def predict_language(text, return_all=False):

    if not is_valid_text(text):
        return {
            "valid": False,
            "language": None,
            "confidence": None,
            "flag": "❌",
            "message": "Invalid input. Please enter meaningful text using real words or sentences in any language.",
            "all_scores": {}
        }

    cleaned, valid = preprocess(text)

    if not valid:
        return {
            "valid": False,
            "language": None,
            "confidence": None,
            "flag": "❌",
            "message": "Invalid text after preprocessing.",
            "all_scores": {}
        }

    model = load_model()

    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_

    idx = proba.argmax()

    language = classes[idx]
    confidence = float(proba[idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "valid": True,
            "language": None,
            "confidence": confidence,
            "flag": "⚠️",
            "message": "Text is ambiguous. Please enter longer text.",
            "all_scores": {}
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
        result["all_scores"] = dict(zip(classes, proba.tolist()))

    return result


if __name__ == "__main__":

    while True:

        text = input("Enter text: ")

        if text.lower() in ["exit", "quit"]:
            break

        result = predict_language(text)

        print(result["message"])