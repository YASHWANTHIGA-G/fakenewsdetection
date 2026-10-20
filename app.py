"""
Simple Fake News Detector - Flask Application.

Provides:
- GET /              : Clean, single-purpose web interface to test news text or headlines.
- POST /api/predict  : JSON text evaluation endpoint (detects Real vs Fake).
- GET /api/model-info: Returns model status and evaluation metrics.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.predict import get_model_info, load_inference_artifacts, predict_news

app = Flask(__name__, template_folder="templates", static_folder="static")

# Warm up models on boot for instantaneous predictions
try:
    load_inference_artifacts()
    predict_news("Washington Senate passed bipartisan infrastructure bill on Tuesday.")
    print("Model artifacts and NLP pipeline successfully preloaded and warmed.")
except Exception as _e:
    print("Model preload notice:", _e)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the simple text-based Fake News Detector page."""
    model_info = get_model_info()
    return render_template("index.html", model_info=model_info)


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Return model status and evaluation metrics."""
    return jsonify(get_model_info())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Direct text submission prediction endpoint.
    Expects JSON: {"text": "news headline or article text"}
    """
    if not request.is_json:
        return jsonify({
            "error": "Invalid request format. Content-Type must be application/json."
        }), 400

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please enter or paste news text to analyze."
        }), 400

    words = text.strip().split()
    if len(words) < 3:
        return jsonify({
            "error": "Please enter at least 3 words (e.g. a complete headline or sentence)."
        }), 400

    char_len = len(text)
    if char_len > 25000:
        return jsonify({
            "error": f"Text exceeds maximum character limit of 25,000 characters."
        }), 400

    try:
        prediction = predict_news(text)
        return jsonify(prediction), 200
    except RuntimeError as e:
        return jsonify({
            "error": str(e)
        }), 503
    except ValueError as e:
        return jsonify({
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "error": f"An unexpected error occurred during prediction: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Fake News Detector running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
