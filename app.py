from flask import Flask, render_template, request, jsonify
import joblib

# Load the saved model and vectorizer
model = joblib.load("disaster_chatbot_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize Flask app
app = Flask(_name_)

@app.route("/")
def home():
    return render_template("index.html")  # HTML frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Expecting JSON input
    message = data.get("message")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Preprocess & predict
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]

    return jsonify({"message": message, "prediction": str(prediction)})

if _name_ == "_main_":
    app.run(debug=True)