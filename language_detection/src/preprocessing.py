"""
preprocessing.py — Text preprocessing for language detection.
Handles cleaning, validation, and basic normalization.
"""

import re
import unicodedata


# Minimum character length for a valid input text
MIN_TEXT_LENGTH = 4

# Regex for detecting "garbage" / random-key input
# If a string is mostly non-letter characters it's likely invalid
_LETTER_RE = re.compile(r'\w', re.UNICODE)


def clean_text(text: str) -> str:
    """
    Light-touch cleaning:
    - Strip leading/trailing whitespace
    - Collapse multiple spaces into one
    - Normalize unicode to NFC (canonical composition)

    We intentionally do NOT remove non-ASCII characters because they
    are vital discriminating features for many languages.
    """
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    return text


def is_valid_text(text: str) -> bool:
    """
    Returns True when the text is long enough and has enough
    alphabetic content to be meaningful.

    Catches:
      - Empty / whitespace-only strings
      - Very short strings (< MIN_TEXT_LENGTH chars)
      - Strings that are mostly numbers / symbols (e.g. random keyboard spam)
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    letters = _LETTER_RE.findall(text)
    if len(letters) < MIN_TEXT_LENGTH:
        return False

    # Require at least 40% of characters to be word characters
    ratio = len(letters) / len(text)
    if ratio < 0.40:
        return False

    return True


def preprocess(text: str):
    """
    Full pipeline: clean → validate.

    Returns:
        (cleaned_text, is_valid)  — if not valid, caller should show
        the 'invalid text' message rather than predicting.
    """
    cleaned = clean_text(text)
    valid = is_valid_text(cleaned)
    return cleaned, valid


if __name__ == "__main__":
    samples = [
        "Hello, how are you?",
        "   ",
        "!!!@@@###",
        "12345",
        "あいうえお",
        "asdfghjkl;",        # random keys — mostly letters but short
        "hhhhhhhh hhhh hhhhh",  # repetitive — still passes (model handles it)
        "Bonjour tout le monde",
    ]
    for s in samples:
        cleaned, valid = preprocess(s)
        print(f"Input: {repr(s):40s} | valid={valid} | cleaned={repr(cleaned)}")
