import pandas as pd
import random
from phonemizer import phonemize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# Sample expected words
words = ["apple", "banana", "cat", "dog", "elephant", "fish", "grape", "hat", "ice", "jungle"]

def introduce_error(phoneme: str, level: str):
    """ Introduce phoneme error based on level. """
    tokens = phoneme.split()
    if level == "almost":
        idx = random.randint(0, len(tokens)-1)
        tokens[idx] = "?"  # minor error
    elif level == "incorrect":
        return "??? ???"
    return " ".join(tokens)

rows = []

for word in words:
    phoneme = phonemize(word, language='en', backend='espeak', strip=True)
    # Correct
    rows.append({"input": phoneme, "label": "correct"})
    # Almost correct (1 phoneme error)
    for _ in range(2):
        noisy = introduce_error(phoneme, "almost")
        rows.append({"input": noisy, "label": "almost"})
    # Incorrect (garbled)
    for _ in range(2):
        noisy = introduce_error(phoneme, "incorrect")
        rows.append({"input": noisy, "label": "incorrect"})

df = pd.DataFrame(rows)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("pronunciation_dataset.csv", index=False)
print("✅ Dataset generated and saved as pronunciation_dataset.csv")

# === Train model ===
X = df["input"]
y = df["label"]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "pronunciation_model.pkl")
print("✅ Model trained and saved as pronunciation_model.pkl")
