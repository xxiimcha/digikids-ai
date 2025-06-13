from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("intent_model.pkl")

@app.route("/predict_intent", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction = model.predict([text])[0]
    return jsonify({"intent": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
