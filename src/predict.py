"""
Verdict - Explainable Prediction & Inference Engine.

Features:
- Lazy loading of model.joblib, vectorizer.joblib, and explainer.joblib with clear error handling.
- predict_news(text, top_n=8) returning:
    - label: "Fake" / "Real"
    - verdict: "Fake" / "Real" / "Unclear" (threshold: confidence < 0.65 yields "Unclear")
    - confidence: float (0.0 - 1.0)
    - fake_probability: float (0.0 - 1.0)
    - real_probability: float (0.0 - 1.0)
    - word_count: int
    - warning: warning if article text is under 40 words
    - signals: list of dicts {word, weight, points_to} derived from:
        contribution = tfidf_value * logistic_regression_coefficient
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.preprocess import clean_text
from src.utils import MODELS_DIR, get_model_probabilities

# Global singletons for lazy-loaded assets
_MODEL: Optional[Any] = None
_VECTORIZER: Optional[Any] = None
_EXPLAINER: Optional[Any] = None
_METRICS: Optional[Dict[str, Any]] = None
_FEATURE_NAMES: Optional[Any] = None
_LR_COEFS: Optional[Any] = None


def load_inference_artifacts() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Lazy-load serialized model, vectorizer, and explainer.
    Pre-warms NLP and cached vocabulary structures for instant inference.
    """
    global _MODEL, _VECTORIZER, _EXPLAINER, _METRICS, _FEATURE_NAMES, _LR_COEFS

    model_path = MODELS_DIR / "model.joblib"
    vec_path = MODELS_DIR / "vectorizer.joblib"
    explainer_path = MODELS_DIR / "explainer.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    if _MODEL is None or _VECTORIZER is None or _EXPLAINER is None:
        missing = []
        if not model_path.exists():
            missing.append(str(model_path))
        if not vec_path.exists():
            missing.append(str(vec_path))
        if not explainer_path.exists():
            missing.append(str(explainer_path))

        if missing:
            raise RuntimeError(
                f"Trained model artifacts not found ({', '.join(missing)}). "
                "Please train the models first by executing: python -m src.train"
            )

        try:
            _MODEL = joblib.load(model_path)
            _VECTORIZER = joblib.load(vec_path)
            _EXPLAINER = joblib.load(explainer_path)
            _FEATURE_NAMES = np.array(_VECTORIZER.get_feature_names_out())
            _LR_COEFS = _EXPLAINER.coef_[0]

            # Pre-warm WordNet Lemmatizer and NLP tokens
            clean_text("warmup text for lemmatizer and instant inference")
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize model artifacts from disk: {e}")

    if _METRICS is None:
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    _METRICS = json.load(f)
            except Exception:
                _METRICS = {}
        else:
            _METRICS = {}

    return _MODEL, _VECTORIZER, _EXPLAINER, _METRICS


def get_model_info() -> Dict[str, Any]:
    """Retrieve active model metadata for frontend display and status checks."""
    try:
        _, _, _, metrics = load_inference_artifacts()
        return {
            "is_trained": True,
            "best_model_name": metrics.get("best_model_name", "Trained Classifier"),
            "test_accuracy": metrics.get("test_accuracy", 0.0),
            "test_f1": metrics.get("test_f1", 0.0),
            "total_articles": metrics.get("total_articles", 0),
            "vocabulary_size": metrics.get("vocabulary_size", 0),
        }
    except RuntimeError:
        return {
            "is_trained": False,
            "best_model_name": "Model Not Yet Trained",
            "test_accuracy": 0.0,
            "test_f1": 0.0,
            "total_articles": 0,
            "vocabulary_size": 0,
        }


def predict_news(text: str, top_n: int = 8) -> Dict[str, Any]:
    """
    Analyze article text and produce an explainable authenticity verdict.

    Parameters:
    ----------
    text : str
        The full text or headline + body of the news article.
    top_n : int
        Number of top influential words to return for attribution.

    Returns:
    -------
    dict:
        label: "Fake" | "Real"
        verdict: "Fake" | "Real" | "Unclear"
        confidence: float in [0.0, 1.0]
        fake_probability: float in [0.0, 1.0]
        real_probability: float in [0.0, 1.0]
        word_count: int
        warning: str | None
        signals: list of dicts [{word, weight, points_to}]
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Article text cannot be empty or solely whitespace.")

    raw_words = text.strip().split()
    word_count = len(raw_words)

    # 1. Ensure models are loaded
    model, vectorizer, explainer, _ = load_inference_artifacts()

    # 2. Preprocess text using symmetric clean_text function
    cleaned = clean_text(text, remove_leakage=True, lemmatize=True, remove_stopwords=True)
    if not cleaned.strip():
        raise ValueError(
            "Article contained only stopwords, numbers, or punctuation. "
            "Insufficient lexical content for analysis."
        )

    # 3. Vectorize text
    vec = vectorizer.transform([cleaned])

    # 4. Compute probabilities
    probs = get_model_probabilities(model, vec)
    fake_prob = float(probs[0])
    # Ensure range [0.0, 1.0]
    fake_prob = max(0.0, min(1.0, fake_prob))
    real_prob = float(1.0 - fake_prob)

    # Base classification
    if fake_prob >= 0.5:
        label = "Fake"
        confidence = fake_prob
    else:
        label = "Real"
        confidence = real_prob

    # Verdict determination: Unclear if confidence < 0.65
    CONFIDENCE_UNCLEAR_THRESHOLD = 0.65
    if confidence < CONFIDENCE_UNCLEAR_THRESHOLD:
        verdict = "Unclear"
    else:
        verdict = label

    # Warning for short articles (under 40 words)
    warning: Optional[str] = None
    if word_count < 40:
        warning = (
            f"This article has only {word_count} words. Short articles lack stylistic depth, "
            "so model confidence may be less reliable."
        )

    # 5. Signal Word Attribution:
    # contribution = tfidf_value * logistic_regression_coefficient
    signals: List[Dict[str, Any]] = []

    try:
        global _FEATURE_NAMES, _LR_COEFS
        feature_names = _FEATURE_NAMES if _FEATURE_NAMES is not None else np.array(vectorizer.get_feature_names_out())
        lr_coefs = _LR_COEFS if _LR_COEFS is not None else explainer.coef_[0]

        # Nonzero entries in the vector for this document
        row = vec.tocoo()
        col_indices = row.col
        tfidf_data = row.data

        if len(col_indices) > 0:
            contributions = []
            for col_idx, tfidf_val in zip(col_indices, tfidf_data):
                word = str(feature_names[col_idx])
                coef = float(lr_coefs[col_idx])
                # Positive pushes to Fake, Negative pushes to Real
                contrib = float(tfidf_val * coef)
                contributions.append((word, contrib, abs(contrib)))

            # Sort by absolute impact
            contributions.sort(key=lambda item: item[2], reverse=True)
            top_contributions = contributions[:top_n]

            # Normalize weights relative to the maximum observed contribution in this doc
            max_abs = top_contributions[0][2] if top_contributions and top_contributions[0][2] > 0 else 1.0

            for word, contrib, abs_contrib in top_contributions:
                normalized_weight = round(float(abs_contrib / max_abs), 3)
                points_to = "fake" if contrib > 0 else "real"
                signals.append(
                    {
                        "word": word,
                        "weight": normalized_weight,
                        "points_to": points_to,
                    }
                )
    except Exception as e:
        # Fallback if explanation fails: return empty signals list
        signals = []

    return {
        "label": label,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "word_count": word_count,
        "warning": warning,
        "signals": signals,
    }


if __name__ == "__main__":
    test_text = (
        "WASHINGTON - The Senate on Thursday passed a major defense spending package "
        "providing funding for personnel salaries, technology development, and international security partnerships."
    )
    print("Testing prediction engine...")
    try:
        res = predict_news(test_text)
        print("Prediction result:", json.dumps(res, indent=2))
    except Exception as e:
        print("Prediction test caught expected exception (models not trained or setup):", e)
