
import joblib
import pandas as pd
import logging
import sys
import time
import uuid

from pathlib import Path
from flask import Flask, jsonify, request, g

# Configure logging to standard output for Render and Docker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "study_hours",
    "attendance",
    "assignments_completed",
    "previous_score"
]


@app.before_request
def start_request_logging():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.start_time = time.perf_counter()


@app.after_request
def finish_request_logging(response):
    request_id = getattr(g, "request_id", str(uuid.uuid4()))
    duration_ms = (time.perf_counter() - getattr(
        g, "start_time", time.perf_counter()
    )) * 1000

    response.headers["X-Request-ID"] = request_id

    level = logging.WARNING if response.status_code >= 400 else logging.INFO

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response


def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file was not found.")

        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH.name)
        return model

    except Exception:
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH.name)
        raise


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
        logger.warning("event=validation_error reason=missing_or_invalid_json")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error missing_fields=%s",
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        logger.info(
            "event=prediction_request feature_names=%s",
            sorted(FEATURES)
        )

        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "PLACED" if prediction_code == 1
            else "NOT PLACED"
        )

        logger.info(
            "event=prediction_success prediction=%s",
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except (TypeError, ValueError):
        logger.warning("event=validation_error reason=non_numeric_features")
        return jsonify({
            "error": "Feature values must be numeric"
        }), 400

    except Exception:
        logger.exception("event=prediction_failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


@app.errorhandler(404)
def not_found(error):
    logger.warning(
        "event=route_not_found path=%s method=%s",
        request.path,
        request.method
    )
    return jsonify({"error": "Route not found"}), 404


if __name__ == "__main__":
    logger.info("event=application_starting port=5000")
    app.run(host="0.0.0.0", port=5000)

