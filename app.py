from flask import Flask, request, jsonify
import whisper
import joblib
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# === Load Whisper model ===
whisper_model = whisper.load_model("base")

# === Train pronunciation classifier from CSV ===
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

# Train the model once at app start
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
    raw_path = f"audio_{timestamp}_{uid}.input"
    wav_path = f"audio_{timestamp}_{uid}.wav"

    try:
        audio.save(raw_path)
        audio_segment = AudioSegment.from_file(raw_path)
        audio_segment.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)

    try:
        result = whisper_model.transcribe(wav_path)
        spoken_word = result['text'].strip().lower()

        # Instead of phonemizer, simulate phoneme with word itself or define a manual phoneme map
        # Here we assume phoneme = spoken_word directly or map manually
        simulated_phoneme = spoken_word  # Or use a lookup if available

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
