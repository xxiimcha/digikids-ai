from flask import Flask, request, jsonify
import whisper
import joblib
import os
import uuid
import subprocess
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Train model from dataset
def train_model_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df["input"]
    y = df["label"]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X, y)
    return pipeline

# Train pronunciation model once at startup
pronunciation_model = train_model_from_csv("pronunciation_dataset.csv")

# Convert to WAV using ffmpeg
def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ], check=True)

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
    raw_path = f"audio_{timestamp}_{uid}.input"
    wav_path = f"audio_{timestamp}_{uid}.wav"

    try:
        audio.save(raw_path)
        convert_to_wav(raw_path, wav_path)
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)

    try:
        result = whisper_model.transcribe(wav_path)
        spoken_word = result['text'].strip().lower()

        simulated_phoneme = spoken_word  # basic assumption for scoring

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
        if os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == '__main__':
    app.run(debug=True)
