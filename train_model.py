# train_model.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Question-to-answer intent data
data = [
    {"text": "what is 2 plus 2", "intent": "math_2_plus_2"},
    {"text": "how much is 2 + 2", "intent": "math_2_plus_2"},
    {"text": "can you solve 2 + 2", "intent": "math_2_plus_2"},

    {"text": "why is the sky blue", "intent": "science_sky_blue"},
    {"text": "tell me why sky is blue", "intent": "science_sky_blue"},
    {"text": "how come sky looks blue", "intent": "science_sky_blue"},

    {"text": "what sound does a dog make", "intent": "animal_sound_dog"},
    {"text": "how does a dog sound", "intent": "animal_sound_dog"},
    {"text": "dog noise", "intent": "animal_sound_dog"},

    {"text": "what sound does a cat make", "intent": "animal_sound_cat"},
    {"text": "how does a cat sound", "intent": "animal_sound_cat"},
    {"text": "meow like a cat", "intent": "animal_sound_cat"},

    {"text": "who are you", "intent": "bot_identity"},
    {"text": "what's your name", "intent": "bot_identity"},
    {"text": "do you have a name", "intent": "bot_identity"},

    {"text": "tell me a fun fact", "intent": "fun_fact"},
    {"text": "say something interesting", "intent": "fun_fact"},
    {"text": "can you tell me something fun", "intent": "fun_fact"},
]

# Extract features and labels
X = [d["text"] for d in data]
y = [d["intent"] for d in data]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Define pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, "intent_model.pkl")
print("Model saved to intent_model.pkl")
