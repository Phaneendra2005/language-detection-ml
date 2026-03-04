"""
app.py — Flask web server for Language Detection System
Run: python app.py
Then open: http://localhost:5000
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Add parent directory so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predict import predict_language

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({
            'valid': False,
            'message': 'Please enter some text.',
            'language': None,
            'confidence': None,
            'flag': '❌',
            'all_scores': {}
        })

    result = predict_language(text, return_all=True)
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Language Detection — Web Interface")
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
