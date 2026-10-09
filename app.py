```python
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "PLACED" if prediction_code == 1
            else "NOT PLACED"
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Feature values must be numeric"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```
