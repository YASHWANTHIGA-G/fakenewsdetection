"""
Verdict - Machine Learning Training & Hyperparameter Tuning Pipeline.

Features:
- 80/20 Stratified train/test split.
- TfidfVectorizer (unigrams + bigrams, max_features=50000, sublinear_tf=True).
- GridSearchCV with 3-fold cross-validation (F1 scoring) across:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Passive Aggressive Classifier
    4. Linear Support Vector Machine (LinearSVC)
    5. Random Forest (optional, skippable via --skip-rf)
- Selects the best overall model by cross-validated F1 score (not test score).
- Full test set evaluation: Accuracy, Precision, Recall, F1, ROC-AUC.
- Explainer model: saves the tuned Logistic Regression for feature attribution.
- Generates reports:
    - reports/model_comparison.csv
    - reports/model_comparison.png
    - reports/confusion_matrices.png
    - reports/roc_curves.png
    - reports/top_features_importance.png
- Saves models/model.joblib, models/vectorizer.joblib, models/explainer.joblib,
  and models/metrics.json.

Usage:
    python -m src.train [--sample N] [--skip-rf] [--force-recompute]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import get_cleaned_dataset
from src.utils import (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_dirs,
    get_model_probabilities,
)

# Visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"Real": "#1E7A4C", "Fake": "#B3261E"}


def define_hyperparameter_grids(skip_rf: bool = False) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    """Define candidate estimators and hyperparameter search grids for 3-fold CV."""
    models: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]},
        ),
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"alpha": [0.01, 0.1, 1.0]},
        ),
        "Passive Aggressive": (
            PassiveAggressiveClassifier(max_iter=1000, random_state=42),
            {"C": [0.1, 0.5, 1.0], "loss": ["hinge", "squared_hinge"]},
        ),
        "Linear SVM": (
            LinearSVC(dual="auto", max_iter=2000, random_state=42),
            {"C": [0.05, 0.5, 1.0]},
        ),
    }

    if not skip_rf:
        models["Random Forest"] = (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100], "max_depth": [20, None]},
        )

    return models


def evaluate_model(
    model: Any,
    X_test_vec: Any,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, f1, and roc_auc for an estimator."""
    y_pred = model.predict(X_test_vec)
    y_probs = get_model_probabilities(model, X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_test, y_probs))
    except Exception:
        auc = 0.5

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc,
    }


def plot_model_comparison_bars(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save multi-metric performance comparison bar chart."""
    df_melted = results_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette="crest",
        edgecolor="#18202E",
        ax=ax,
    )

    ax.set_title("Cross-Model Performance Benchmarking", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_xlabel("Candidate Model", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_file = output_dir / "model_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_confusion_matrices(
    fitted_models: Dict[str, Any],
    X_test_vec: Any,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot and save confusion matrices for all evaluated models in a grid layout."""
    n_models = len(fitted_models)
    cols = min(3, n_models)
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=300)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, model) in enumerate(fitted_models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_test_vec)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Real (0)", "Fake (1)"],
            yticklabels=["Real (0)", "Fake (1)"],
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    # Hide unused subplots if grid has empty slots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Confusion Matrix Grid on Test Set", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_file = output_dir / "confusion_matrices.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_roc_curves(
    fitted_models: Dict[str, Any],
    X_test_vec: Any,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot and save combined ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for name, model in fitted_models.items():
        y_probs = get_model_probabilities(model, X_test_vec)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        try:
            auc_val = roc_auc_score(y_test, y_probs)
        except Exception:
            auc_val = 0.5
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    out_file = output_dir / "roc_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def extract_and_plot_top_features(
    explainer_lr: LogisticRegression,
    vectorizer: TfidfVectorizer,
    output_dir: Path,
    top_n: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract top 20 words predicting Fake and Real news based on Logistic Regression coefficients,
    and save a dual horizontal bar chart.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = explainer_lr.coef_[0]

    # Positive coefficients indicate class 1 (Fake), Negative coefficients indicate class 0 (Real)
    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    fake_features = [
        {"word": feature_names[i], "coefficient": float(coefs[i])}
        for i in top_fake_idx
    ]
    real_features = [
        {"word": feature_names[i], "coefficient": float(abs(coefs[i]))}
        for i in top_real_idx
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300)

    # Plot Real features
    r_words = [f["word"] for f in reversed(real_features)]
    r_vals = [f["coefficient"] for f in reversed(real_features)]
    ax1.barh(r_words, r_vals, color=PALETTE["Real"], edgecolor="#18202E")
    ax1.set_title(f"Top {top_n} Tokens Indicating REAL News", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Absolute Logistic Coefficient Weight", fontsize=10)

    # Plot Fake features
    f_words = [f["word"] for f in reversed(fake_features)]
    f_vals = [f["coefficient"] for f in reversed(fake_features)]
    ax2.barh(f_words, f_vals, color=PALETTE["Fake"], edgecolor="#18202E")
    ax2.set_title(f"Top {top_n} Tokens Indicating FAKE News", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Positive Logistic Coefficient Weight", fontsize=10)

    plt.suptitle("Logistic Regression Feature Importance Spectrum", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_file = output_dir / "top_features_importance.png"
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

    return {"top_fake_words": fake_features, "top_real_words": real_features}


def train_models(
    sample_size: int | None = None,
    skip_rf: bool = False,
    force_recompute: bool = False,
) -> None:
    """Execute complete training, grid search, evaluation, and artifact serialization."""
    ensure_dirs()
    print("=" * 60)
    print("VERDICT - TRAINING AND EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load Cleaned Dataset
    print("\n[1/6] Loading cleaned text dataset...")
    df = get_cleaned_dataset(
        force_recompute=force_recompute,
        sample_size=sample_size,
    )
    total_articles = len(df)
    print(f"Articles loaded: {total_articles} (Real: {(df['label'] == 0).sum()}, Fake: {(df['label'] == 1).sum()})")

    # 2. Stratified Train/Test Split (80/20)
    print("\n[2/6] Splitting dataset (80% train, 20% test stratified)...")
    X = df["cleaned_text"].values
    y = df["label"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"Training set: {len(X_train_raw)} | Test set: {len(X_test_raw)}")

    # 3. TF-IDF Feature Extraction
    print("\n[3/6] Fitting TfidfVectorizer (unigrams + bigrams, sublinear_tf=True)...")
    min_df_val = 2 if len(X_train_raw) > 50 else 1
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=min_df_val,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"Extracted vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # 4. Model Training & GridSearchCV
    print("\n[4/6] Running 3-fold Stratified GridSearchCV for candidate classifiers...")
    candidate_configs = define_hyperparameter_grids(skip_rf=skip_rf)
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fitted_best_models: Dict[str, Any] = {}
    cv_f1_scores: Dict[str, float] = {}
    eval_records: List[Dict[str, Any]] = []

    explainer_lr: LogisticRegression | None = None

    for model_name, (estimator, param_grid) in candidate_configs.items():
        print(f"  --> Tuning {model_name}...")
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1",
            cv=cv_strategy,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train_vec, y_train)

        best_estimator = grid.best_estimator_
        best_cv_f1 = float(grid.best_score_)

        fitted_best_models[model_name] = best_estimator
        cv_f1_scores[model_name] = best_cv_f1

        if model_name == "Logistic Regression":
            explainer_lr = best_estimator

        # Evaluate on the hold-out test set
        metrics = evaluate_model(best_estimator, X_test_vec, y_test)
        record = {
            "Model": model_name,
            "CV_F1": best_cv_f1,
            "Best_Params": str(grid.best_params_),
            **metrics,
        }
        eval_records.append(record)
        print(f"      Best CV F1: {best_cv_f1:.4f} | Test F1: {metrics['F1 Score']:.4f} | Test Acc: {metrics['Accuracy']:.4f}")

    # Pick the best model based on CROSS-VALIDATED F1 score (not test score, to prevent overfitting)
    best_model_name = max(cv_f1_scores, key=cv_f1_scores.get)
    best_model = fitted_best_models[best_model_name]
    print(f"\nOptimal Model by 3-Fold CV F1: >>> {best_model_name} <<< (CV F1 = {cv_f1_scores[best_model_name]:.4f})")

    # If explainer was not Logistic Regression (should never happen), ensure explainer_lr is fitted
    if explainer_lr is None:
        explainer_lr = LogisticRegression(max_iter=1000, random_state=42)
        explainer_lr.fit(X_train_vec, y_train)

    # 5. Serialization and Evaluation Artifacts
    print("\n[5/6] Generating comparison charts, confusion matrices, and ROC curves...")
    results_df = pd.DataFrame(eval_records).sort_values(by="CV_F1", ascending=False).reset_index(drop=True)
    comparison_csv_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved: {comparison_csv_path}")

    plot_model_comparison_bars(results_df, REPORTS_DIR)
    plot_confusion_matrices(fitted_best_models, X_test_vec, y_test, REPORTS_DIR)
    plot_roc_curves(fitted_best_models, X_test_vec, y_test, REPORTS_DIR)

    top_words_data = extract_and_plot_top_features(explainer_lr, vectorizer, REPORTS_DIR)

    # 6. Save models and metrics.json
    print("\n[6/6] Serializing production models and metrics metadata...")
    model_path = MODELS_DIR / "model.joblib"
    vec_path = MODELS_DIR / "vectorizer.joblib"
    explainer_path = MODELS_DIR / "explainer.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)
    joblib.dump(explainer_lr, explainer_path)

    # Extract test accuracy of the best model
    best_model_test_metrics = next(
        r for r in eval_records if r["Model"] == best_model_name
    )

    metrics_payload = {
        "best_model_name": best_model_name,
        "best_cv_f1": cv_f1_scores[best_model_name],
        "test_accuracy": best_model_test_metrics["Accuracy"],
        "test_f1": best_model_test_metrics["F1 Score"],
        "test_precision": best_model_test_metrics["Precision"],
        "test_recall": best_model_test_metrics["Recall"],
        "test_roc_auc": best_model_test_metrics["ROC-AUC"],
        "total_articles": total_articles,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "all_models_benchmarked": eval_records,
        "top_features": top_words_data,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved: {model_path}")
    print(f"Saved: {vec_path}")
    print(f"Saved: {explainer_path}")
    print(f"Saved: {metrics_path}")
    print("\nTraining completed successfully! Pipeline is ready for inference.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark fake news detection models.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows from the dataset for rapid verification and debugging.",
    )
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip Random Forest training to accelerate iteration time.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing the cleaned dataset cache even if present.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(
        sample_size=args.sample,
        skip_rf=args.skip_rf,
        force_recompute=args.force_recompute,
    )
