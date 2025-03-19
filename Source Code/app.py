from flask import Flask, request, render_template, jsonify
import re
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

model = load_model('sarcasm_sentiment_model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

sarcasm_classes = {0: 'Not Sarcastic', 1: 'Sarcastic'}
sentiment_classes = {0: 'Negative Sentiment', 1: 'Positive Sentiment'}

def is_valid_text(text):
    """Checks if the text contains only letters, spaces, and certain punctuation."""
    return bool(re.fullmatch(r"[A-Za-z !?,.:;\-'\"()]+", text))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text')

    if not input_text:
        return jsonify({"error": "Invalid input"}), 400
    
    if not is_valid_text(input_text):
        return jsonify({"error": "Text contains invalid characters. Only letters and emotion-related symbols are allowed."}), 400
    
    try:
        processed_text = tfidf_vectorizer.transform([input_text]).toarray()
        predictions = model.predict(processed_text)
        sarcasm_prediction = int(predictions[0][0] > 0.5)
        sentiment_prediction = int(predictions[0][1] > 0.5)

        return jsonify({
            "sarcasm_prediction": sarcasm_classes[sarcasm_prediction],
            "sentiment_prediction": sentiment_classes[sentiment_prediction]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
