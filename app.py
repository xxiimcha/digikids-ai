from flask import Flask, request, jsonify
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import whisper

app = Flask(__name__)

# === Train pronunciation model once (light) ===
def train_model_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df["input"]
    y = df["label"]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100)),  # limit features
        ('clf', LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X, y)
    return pipeline

pronunciation_model = train_model_from_csv("pronunciation_dataset.csv")

@app.route('/api/pronunciation-feedback', methods=['POST'])
def pronunciation_feedback():
    if 'audio' not in request.files or 'expected' not in request.form:
        return jsonify({"error": "Missing audio or expected field"}), 400

    audio = request.files['audio']
    expected_word = request.form['expected'].strip().lower()

    if audio.filename == '':
        return jsonify({"error": "No file selected"}), 400

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uid = str(uuid.uuid4())[:8]
    audio_path = f"temp_{timestamp}_{uid}.{audio.filename.split('.')[-1]}"
    audio.save(audio_path)

    try:
        # Load Whisper only when needed
        whisper_model = whisper.load_model("tiny")  # or "base.en" if supported
        result = whisper_model.transcribe(audio_path)
        spoken_word = result['text'].strip().lower()

        simulated_phoneme = spoken_word
        prediction = pronunciation_model.predict([simulated_phoneme])[0]
        proba = pronunciation_model.predict_proba([simulated_phoneme])[0].max()

        feedback = {
            "correct": "Great job! You said it correctly.",
            "almost": "Almost there! Let's try again.",
            "incorrect": "Let's try saying it again together!"
        }.get(prediction, "Hmm, I’m not sure.")

        return jsonify({
            "spoken_word": spoken_word,
            "expected_word": expected_word,
            "simulated_phoneme": simulated_phoneme,
            "prediction": prediction,
            "confidence": round(float(proba), 2),
            "feedback": feedback
        })
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
