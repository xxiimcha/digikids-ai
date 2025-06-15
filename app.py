from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained intent classification model
model = joblib.load("intent_model.pkl")

# Define a dictionary that maps intents to answers
intent_to_answer = {
    "math_2_plus_2": "2 plus 2 is 4!",
    "science_sky_blue": "The sky looks blue because of how sunlight scatters in the atmosphere.",
    "animal_sound_dog": "A dog says woof woof!",
    "animal_sound_cat": "A cat says meow meow!",
    "bot_identity": "I'm your friendly fox assistant!",
    "fun_fact": "Did you know that a group of flamingos is called a flamboyance?",
}

@app.route("/predict_intent", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Predict the intent
        prediction = model.predict([text])[0]
        answer = intent_to_answer.get(prediction, "Hmm, I don't know the answer to that yet.")
        return jsonify({"intent": prediction, "answer": answer})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
