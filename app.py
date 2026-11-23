from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

model = joblib.load('model/diabetes_model.pkl')

columns = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Diabetes AI API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not all(c in data for c in columns):
        return jsonify({"error": "Provide all 8 patient features"}), 400

    try:
        patient = pd.DataFrame(
            [[float(data[c]) for c in columns]],
            columns=columns
        )

        if patient.isnull().values.any():
            return jsonify({"error": "Invalid input values"}), 400

        prediction = int(model.predict(patient)[0])

        return jsonify({
            "prediction": prediction,
            "result": (
                "Higher likelihood of diabetes"
                if prediction == 1
                else "Lower likelihood of diabetes"
            )
        })

    except (ValueError, TypeError):
        return jsonify({"error": "All features must be numeric"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)