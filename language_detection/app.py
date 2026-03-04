"""
app.py — Flask web server for Language Detection System
"""

import os
import sys
import re
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import predict_language

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def valid_input(text):

    if not text.strip():
        return False

    pattern = r"[A-Za-z\u0C00-\u0C7F\u0600-\u06FF\u4E00-\u9FFF\u0900-\u097F]"
    return bool(re.search(pattern, text))


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "valid": False,
            "message": "Please enter some text.",
            "language": None,
            "confidence": None,
            "flag": "❌",
            "all_scores": {}
        })

    if not valid_input(text):
        return jsonify({
            "valid": False,
            "message": "Please enter valid text containing real words.",
            "language": None,
            "confidence": None,
            "flag": "❌",
            "all_scores": {}
        })

    result = predict_language(text, return_all=True)

    return jsonify(result)


if __name__ == "__main__":

    print("\n" + "="*50)
    print("  Language Detection — Web Interface")
    print("  Server starting...")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=10000)