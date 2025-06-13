# train_model.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Training data
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

X = [d["text"] for d in data]
y = [d["intent"] for d in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "intent_model.pkl")
