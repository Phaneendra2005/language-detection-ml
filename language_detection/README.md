# 🌐 Language Detection System

Detect the language of any sentence using **character n-gram TF-IDF features** and a **LinearSVC** classifier. Supports **21 languages** with high accuracy.

---

## 📁 Project Structure

```
language_detection/
├── data/
│   ├── generate_dataset.py   # Generates the training dataset
│   └── dataset.csv           # Created automatically on first train
├── models/
│   ├── language_model.pkl    # Trained model (created by train.py)
│   └── languages.txt         # List of supported languages
├── src/
│   ├── preprocessing.py      # Text cleaning + invalid text detection
│   └── feature_extraction.py # TF-IDF character n-gram vectorizer
├── train.py                  # Train the model
├── predict.py                # Predict language (CLI + interactive)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Auto-generate the dataset if not present
- Train TF-IDF + LinearSVC pipeline
- Print accuracy report
- Save model to `models/language_model.pkl`

### 3. Predict
```bash
# Interactive mode
python predict.py

# Single prediction
python predict.py --text "Bonjour tout le monde"

# With confidence scores
python predict.py --text "こんにちは、元気ですか" --verbose

# Batch file
python predict.py --file my_sentences.txt
```

---

## 🌍 Supported Languages

| Language   | Script   | Language   | Script   |
|------------|----------|------------|----------|
| English    | Latin    | Hindi      | Devanagari |
| French     | Latin    | Dutch      | Latin    |
| Spanish    | Latin    | Swedish    | Latin    |
| German     | Latin    | Turkish    | Latin    |
| Italian    | Latin    | Polish     | Latin    |
| Portuguese | Latin    | Vietnamese | Latin    |
| Russian    | Cyrillic | Tamil      | Tamil    |
| Arabic     | Arabic   | Urdu       | Arabic   |
| Chinese    | Han      | Greek      | Greek    |
| Japanese   | CJK Mix  | Danish     | Latin    |
| Korean     | Hangul   |            |          |

---

## ⚙️ How It Works

```
Input Text
    │
    ▼
┌─────────────────────────┐
│   Preprocessing          │  ← Clean, validate, detect garbage input
└─────────────────────────┘
    │ invalid? → "Invalid text" message
    ▼
┌─────────────────────────┐
│  TF-IDF (char 2-4 grams)│  ← Extract character n-gram features
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LinearSVC Classifier    │  ← Calibrated for probability output
└─────────────────────────┘
    │
    ▼
Language + Confidence Score
```

### Why Character N-grams?
- Works for ALL scripts (Latin, Cyrillic, CJK, Arabic, Devanagari …)
- No tokenization needed
- Language-specific patterns (e.g., `sch`, `tion`, `ли`) are naturally captured
- Robust to unknown words and typos

### Why LinearSVC?
- 97-100% accuracy on language identification
- Orders of magnitude faster than neural models for this task
- `CalibratedClassifierCV` adds probability scores for confidence display

---

## 🧪 Example Predictions

```
"Hello, how are you today?"             → 🇬🇧 English    (99.2%)
"Bonjour, comment allez-vous?"          → 🇫🇷 French     (98.7%)
"Hola, ¿cómo estás hoy?"               → 🇪🇸 Spanish    (97.5%)
"こんにちは、元気ですか？"                → 🇯🇵 Japanese   (99.8%)
"مرحباً، كيف حالك اليوم؟"              → 🇸🇦 Arabic     (98.1%)
"!@#$%^&*()"                           → ❌ Invalid text
"asdf"                                  → ❌ Invalid text
"12345"                                 → ❌ Invalid text
```

---

## 📊 Using Your Own Kaggle Dataset

The model is compatible with the popular Kaggle **Language Detection** dataset.
Download it and point train.py to your CSV:

```bash
python train.py --dataset /path/to/Language Detection.csv
```

The CSV must have columns `Text` and `Language` (or `text` and `language`).

---

## 📦 Requirements

- Python 3.8+
- scikit-learn ≥ 1.0
- pandas ≥ 1.3
- numpy ≥ 1.21
