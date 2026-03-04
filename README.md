# Language Detection System using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-black)
![Status](https://img.shields.io/badge/Project-Completed-green)

A Machine Learning based Language Detection System that predicts the language of input text using NLP preprocessing and classification techniques.  
The system is trained on multilingual datasets including Telugu and provides predictions through both scripts and a web interface.

---

# Project Overview

Language detection is a fundamental Natural Language Processing (NLP) task used in search engines, translation systems, and chat applications.

This project implements a machine learning pipeline that:

1. Cleans and preprocesses multilingual text
2. Converts text into numerical features
3. Trains a classification model
4. Predicts the language of new input text

The system can identify multiple languages including Telugu with high accuracy.

---

# System Architecture

```
User Input
    ↓
Text Preprocessing
    ↓
Feature Extraction (Vectorization)
    ↓
Machine Learning Model
    ↓
Language Prediction
```

---

# Features

- Multilingual language detection
- Machine learning based classification
- Telugu language dataset support
- Flask web interface
- Fast prediction using trained models
- Modular project structure

---

# Technologies Used

| Category | Technology |
|--------|-----------|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Frontend | HTML |
| Version Control | Git, GitHub |

---

# Project Structure

```
language_detection/
│
├── app.py                  # Flask web application
├── train.py                # Model training script
├── predict.py              # Prediction script
│
├── data/
│   ├── dataset.csv
│   └── generate_dataset.py
│
├── models/
│   └── languages.txt
│
├── src/
│   ├── preprocessing.py
│   └── feature_extraction.py
│
├── templates/
│   └── index.html
│
└── requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Phaneendra2005/language-detection-ml.git
```

Move to the project directory

```bash
cd language-detection-ml
```

Install required dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

### Train the Model

```bash
python train.py
```

### Run Prediction Script

```bash
python predict.py
```

### Run Web Application

```bash
python app.py
```

Open in browser

```
http://127.0.0.1:5000
```

---

# Example

Input Text

```
నాకు తెలుగు చాలా ఇష్టం
```

Prediction

```
Telugu
```

---

# Model Workflow

1. Dataset collection
2. Text preprocessing
3. Feature extraction
4. Model training
5. Model evaluation
6. Language prediction

---

# Future Improvements

- Deep Learning models (LSTM / Transformers)
- Larger multilingual dataset
- REST API for integration
- Cloud deployment
- Real-time language detection service

---

# Author

Phaneendra  
Computer Science Student  
Specialization: Cybersecurity

---

# License

This project is open-source and available under the MIT License.
