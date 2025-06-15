from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained intent classification model
model = joblib.load("intent_model.pkl")

# Confidence threshold for accepting a prediction
CONFIDENCE_THRESHOLD = 0.6

# Mapping from intent labels to response messages
intent_to_answer = {
    "math_2_plus_2": "2 plus 2 is 4!",
    "science_sky_blue": "The sky looks blue because of how sunlight scatters in the atmosphere.",
    "animal_sound_dog": "A dog says woof woof!",
    "animal_sound_cat": "A cat says meow meow!",
    "bot_identity": "I'm your friendly fox assistant!",
    "fun_fact": "Did you know that a group of flamingos is called a flamboyance?",
}

@app.route("/predict_intent", methods=["POST"])
def predict_intent():
    data = request.get_json()

    if not data or not data.get("text"):
        return jsonify({
            "intent": "unknown",
            "answer": "I didn’t hear anything. Can you say that again?"
        })

    user_input = data["text"].strip()

    try:
        # Get prediction probabilities
        probabilities = model.predict_proba([user_input])[0]
        max_index = probabilities.argmax()
        max_confidence = probabilities[max_index]
        predicted_intent = model.classes_[max_index]

        # Check if confidence is too low
        if max_confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "intent": "unknown",
                "answer": "I don't understand. Can you try again?"
            })

        # Fetch corresponding answer
        answer = intent_to_answer.get(predicted_intent, "I don't understand. Can you try again?")
        return jsonify({
            "intent": predicted_intent,
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "intent": "error",
            "answer": "Oops! Something went wrong. Please try again.",
            "error": str(e)  # Optional: remove in production
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
