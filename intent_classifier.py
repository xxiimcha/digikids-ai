
import json
import joblib
from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample training data
data = [
    {"text": "show me numbers", "intent": "open_numbers"},
    {"text": "open the number page", "intent": "open_numbers"},
    {"text": "I want to see numbers", "intent": "open_numbers"},
    {"text": "open shapes", "intent": "open_shapes"},
    {"text": "go to shapes", "intent": "open_shapes"},
    {"text": "I want shapes", "intent": "open_shapes"},
    {"text": "what's your name", "intent": "bot_name"},
    {"text": "who are you", "intent": "bot_name"},
    {"text": "tell me a joke", "intent": "tell_joke"},
    {"text": "make me laugh", "intent": "tell_joke"},
    {"text": "go home", "intent": "go_home"},
    {"text": "back to main", "intent": "go_home"},
    {"text": "start learning", "intent": "start_learning"},
    {"text": "let's begin", "intent": "start_learning"},
]

# Prepare training data
X = [item["text"] for item in data]
y = [item["intent"] for item in data]

# Split for potential testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'intent_model.pkl')

# Set up Flask API
app = Flask(__name__)
model = joblib.load('intent_model.pkl')

@app.route('/predict_intent', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        prediction = model.predict([text])[0]
        return jsonify({"intent": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)