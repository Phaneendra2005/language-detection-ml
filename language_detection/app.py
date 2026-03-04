import os
import sys
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import predict_language, is_valid_text

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


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

    if not is_valid_text(text):
        return jsonify({
            "valid": False,
            "message": "Invalid text. Please enter meaningful words or sentences.",
            "language": None,
            "confidence": None,
            "flag": "❌",
            "all_scores": {}
        })

    result = predict_language(text, return_all=True)

    return jsonify(result)


if __name__ == "__main__":

    print("\n" + "="*50)
    print("Language Detection — Web Interface")
    print("Server starting...")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=10000)