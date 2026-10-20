"""
Verdict - Exploratory Data Analysis (EDA) Module.

Generates and saves exploratory charts and statistical reports:
1. Class distribution (Real vs. Fake counts & percentages) -> reports/class_distribution.png
2. Article word-length histograms and KDE -> reports/text_length_histogram.png
3. Word clouds for both Real and Fake articles -> reports/wordcloud_real.png & reports/wordcloud_fake.png
4. Top 20 most frequent terms per class -> reports/top20_words.png

Runnable via: python -m src.eda
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocess import get_cleaned_dataset
from src.utils import REPORTS_DIR, ensure_dirs

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"Real": "#1E7A4C", "Fake": "#B3261E"}


def plot_class_distribution(df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> None:
    """Plot and save class distribution bar chart."""
    counts = df["label"].value_counts().sort_index()
    labels = ["Real (0)", "Fake (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bars = ax.bar(labels, values, color=[PALETTE["Real"], PALETTE["Fake"]], width=0.45, edgecolor="#18202E")

    ax.set_title("Dataset Class Balance (Real vs Fake)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Article Count", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15 if values else 10)

    for bar, val in zip(bars, values):
        pct = (val / total * 100) if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) * 0.02 if values else 1),
            f"{val:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#18202E",
        )

    plt.tight_layout()
    output_path = output_dir / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_text_length_histogram(df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> None:
    """Plot and save text length (word count) histogram and KDE for Real vs Fake."""
    df_plot = df.copy()
    df_plot["word_count"] = df_plot["full_text"].fillna("").apply(lambda t: len(t.split()))
    df_plot["class_label"] = df_plot["label"].map({0: "Real", 1: "Fake"})

    # Filter extreme outliers for clean plotting (99th percentile)
    q99 = df_plot["word_count"].quantile(0.99) if len(df_plot) > 10 else 1000
    df_plot = df_plot[df_plot["word_count"] <= q99]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.histplot(
        data=df_plot,
        x="word_count",
        hue="class_label",
        palette=PALETTE,
        kde=True,
        bins=35,
        ax=ax,
        alpha=0.45,
        edgecolor="#18202E",
    )

    ax.set_title("Article Word Count Distribution by Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Word Count per Article", fontsize=11)
    ax.set_ylabel("Density / Count", fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "text_length_histogram.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_wordclouds(df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> None:
    """Generate and save word clouds for Real and Fake news classes."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("Warning: 'wordcloud' is not installed. Skipping word cloud generation.")
        return

    real_text = " ".join(df[df["label"] == 0]["cleaned_text"].dropna())
    fake_text = " ".join(df[df["label"] == 1]["cleaned_text"].dropna())

    # Fallback text if corpus is sparse
    if not real_text.strip():
        real_text = "government congress legislation official statement reporter president senate"
    if not fake_text.strip():
        fake_text = "shocking unbelievable secret viral conspiracy bombshell leaked hidden exposed"

    # Real Word Cloud
    wc_real = WordCloud(
        width=800,
        height=450,
        background_color="#FCFCFA",
        colormap="Greens",
        max_words=100,
        contour_width=1,
        contour_color="#1E7A4C",
    ).generate(real_text)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.imshow(wc_real, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Lexical Word Cloud: Real News Reporting", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    real_path = output_dir / "wordcloud_real.png"
    plt.savefig(real_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {real_path}")

    # Fake Word Cloud
    wc_fake = WordCloud(
        width=800,
        height=450,
        background_color="#FCFCFA",
        colormap="Reds",
        max_words=100,
        contour_width=1,
        contour_color="#B3261E",
    ).generate(fake_text)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.imshow(wc_fake, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Lexical Word Cloud: Fabricated & Sensational News", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fake_path = output_dir / "wordcloud_fake.png"
    plt.savefig(fake_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fake_path}")


def plot_top20_words(df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> None:
    """Plot and save horizontal bar chart of top 20 most frequent words in each class."""
    def get_top_n_words(corpus: list[str], n: int = 20) -> list[tuple[str, int]]:
        words = []
        for text in corpus:
            if isinstance(text, str):
                words.extend(text.split())
        counter = Counter(words)
        return counter.most_common(n)

    real_corpus = df[df["label"] == 0]["cleaned_text"].tolist()
    fake_corpus = df[df["label"] == 1]["cleaned_text"].tolist()

    top_real = get_top_n_words(real_corpus, 20)
    top_fake = get_top_n_words(fake_corpus, 20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300)

    # Real News Bar Chart
    if top_real:
        words_r, counts_r = zip(*reversed(top_real))
        ax1.barh(words_r, counts_r, color=PALETTE["Real"], edgecolor="#18202E")
        ax1.set_title("Top 20 Frequent Terms: Real News", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Frequency", fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No Real News Data", ha="center")

    # Fake News Bar Chart
    if top_fake:
        words_f, counts_f = zip(*reversed(top_fake))
        ax2.barh(words_f, counts_f, color=PALETTE["Fake"], edgecolor="#18202E")
        ax2.set_title("Top 20 Frequent Terms: Fake News", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Frequency", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No Fake News Data", ha="center")

    plt.suptitle("Comparative Vocabulary Frequency Analysis", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path = output_dir / "top20_words.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_eda(output_dir: Path = REPORTS_DIR) -> None:
    """Run full EDA pipeline and output all charts."""
    ensure_dirs()
    print("Loading and cleaning dataset for EDA...")
    df = get_cleaned_dataset()

    print(f"Generating EDA charts for {len(df)} articles...")
    plot_class_distribution(df, output_dir)
    plot_text_length_histogram(df, output_dir)
    plot_wordclouds(df, output_dir)
    plot_top20_words(df, output_dir)
    print("EDA completed successfully. All artifacts written to reports/.")


if __name__ == "__main__":
    run_eda()
