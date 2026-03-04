"""
predict.py — Predict the language of a given text.

Features:
  - Loads the trained pipeline model
  - Validates input (shows "Invalid text" for garbage/random input)
  - Returns language name + confidence score
  - Can be used as a module or run interactively from the CLI

Usage:
    python predict.py                        # interactive REPL
    python predict.py --text "Hello world"   # single prediction
    python predict.py --file texts.txt       # predict each line of a file
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "language_model.pkl")

# Minimum confidence to trust a prediction
# If the top-class probability is below this we also flag as uncertain
CONFIDENCE_THRESHOLD = 0.30

# Emoji flags for a nicer display (extended mapping)
LANGUAGE_FLAGS = {
    "English":    "🇬🇧",
    "French":     "🇫🇷",
    "Spanish":    "🇪🇸",
    "German":     "🇩🇪",
    "Italian":    "🇮🇹",
    "Portuguese": "🇵🇹",
    "Russian":    "🇷🇺",
    "Arabic":     "🇸🇦",
    "Chinese":    "🇨🇳",
    "Japanese":   "🇯🇵",
    "Korean":     "🇰🇷",
    "Hindi":      "🇮🇳",
    "Dutch":      "🇳🇱",
    "Swedish":    "🇸🇪",
    "Turkish":    "🇹🇷",
    "Polish":     "🇵🇱",
    "Vietnamese": "🇻🇳",
    "Tamil":      "🇮🇳",
    "Urdu":       "🇵🇰",
    "Greek":      "🇬🇷",
    "Danish":     "🇩🇰",
    "Telugu":     "🇮🇳",
}


# ─── Model loading ────────────────────────────────────────────────────────────

_cached_model = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the trained sklearn Pipeline (cached after first load)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Run  python train.py  first to train and save the model."
        )
    with open(model_path, "rb") as f:
        _cached_model = pickle.load(f)
    return _cached_model


# ─── Core prediction ──────────────────────────────────────────────────────────

def predict_language(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    return_all: bool = False,
) -> dict:
    """
    Predict the language of `text`.

    Returns a dict:
        {
            "valid":       bool,
            "language":    str | None,
            "confidence":  float | None,   # 0–1
            "flag":        str,
            "message":     str,
            "all_scores":  dict[str, float]  # only when return_all=True
        }
    """
    # ── Step 1: Preprocess & validate ─────────────────────────────────────────
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

    # ── Step 2: Load model & predict ──────────────────────────────────────────
    model = load_model(model_path)
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_

    top_idx = proba.argmax()
    language = classes[top_idx]
    confidence = float(proba[top_idx])

    # ── Step 3: Low-confidence guard ──────────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "valid": True,
            "language": None,
            "confidence": confidence,
            "flag": "⚠️",
            "message": (
                f"Text is ambiguous (max confidence {confidence*100:.1f}%). "
                "Try a longer or more typical sentence."
            ),
            "all_scores": dict(zip(classes, proba.tolist())) if return_all else {},
        }

    flag = LANGUAGE_FLAGS.get(language, "🌐")
    result = {
        "valid": True,
        "language": language,
        "confidence": confidence,
        "flag": flag,
        "message": f"{flag}  Detected Language: {language}  ({confidence*100:.1f}% confidence)",
    }
    if return_all:
        # Sort by probability descending
        result["all_scores"] = dict(
            sorted(zip(classes, proba.tolist()), key=lambda x: x[1], reverse=True)
        )
    else:
        result["all_scores"] = {}

    return result


# ─── Pretty printer ───────────────────────────────────────────────────────────

def print_result(result: dict, verbose: bool = False) -> None:
    print(f"\n  {result['message']}")
    if verbose and result.get("all_scores"):
        print("\n  Top-5 probabilities:")
        for i, (lang, score) in enumerate(list(result["all_scores"].items())[:5]):
            bar = "█" * int(score * 30)
            print(f"    {lang:15s} {score*100:5.1f}%  {bar}")
    print()


# ─── Interactive REPL ─────────────────────────────────────────────────────────

def interactive_mode(model_path: str = DEFAULT_MODEL_PATH) -> None:
    print("\n" + "=" * 55)
    print("  Language Detection System")
    print("  Type a sentence → get its language")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 55)

    try:
        load_model(model_path)
        print("  Model loaded successfully ✓\n")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    while True:
        try:
            text = input("  Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        if not text:
            continue

        result = predict_language(text, model_path, return_all=True)
        print_result(result, verbose=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the language of text")
    parser.add_argument("--text", "-t", type=str, help="Text to classify")
    parser.add_argument(
        "--file", "-f", type=str,
        help="Path to a text file — predict language for each non-empty line"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show top-5 probability scores"
    )
    args = parser.parse_args()

    if args.text:
        result = predict_language(args.text, args.model, return_all=args.verbose)
        print_result(result, verbose=args.verbose)

    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        print(f"\nPredicting {len(lines)} lines from '{args.file}':\n")
        for line in lines:
            result = predict_language(line, args.model, return_all=args.verbose)
            print(f"  {line[:60]:60s} → {result['message']}")

    else:
        interactive_mode(args.model)
