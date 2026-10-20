"""
Verdict - Text Preprocessing & Leakage Removal Pipeline.

Features:
- NLTK auto-downloader for required models (punkt, wordnet, stopwords).
- Regex-based stripping of datelines (e.g., 'WASHINGTON (Reuters) -'),
  source watermarks ('getty images', '21st century wire', 'pic.twitter.com'),
  URLs, HTML tags, and numerical artifacts.
- Lemmatization and stopword removal.
- Shared clean_text() function used symmetrically in training, evaluation,
  and real-time inference.
- Dataset caching to avoid redundant text processing.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.utils import CLEANED_CSV_PATH, DATA_DIR, ensure_dirs, load_raw_dataset

# Instant stopwords and lemmatizer initialization without network calls
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = ENGLISH_STOP_WORDS

try:
    from nltk.stem import WordNetLemmatizer
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None

# Precompiled regex patterns for maximum processing speed
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
TWITTER_PIC_PATTERN = re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE)
TWITTER_USER_PATTERN = re.compile(r"@\w+")

# Data leakage patterns commonly found in the ISOT dataset:
# - Dateline formats: e.g. "WASHINGTON (Reuters) -", "LONDON (Reuters) --", "[City] (Reuters) - "
DATELINE_PATTERN = re.compile(
    r"^[\s\w\.,\-\/]+?\([a-zA-Z\s]+\)\s*[-–—]+\s*",
    re.IGNORECASE,
)
REUTERS_PATTERN = re.compile(r"\b(reuters)\b", re.IGNORECASE)
WATERMARKS_PATTERN = re.compile(
    r"\b(featured image|image via|getty images|21st century wire|wire|via getty|getty)\b",
    re.IGNORECASE,
)
NUMBERS_PATTERN = re.compile(r"\b\d+\b")
PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(
    text: str,
    remove_leakage: bool = True,
    lemmatize: bool = True,
    remove_stopwords: bool = True,
) -> str:
    """
    Standardize, sanitize, and normalize text for feature extraction and inference.

    Parameters:
    ----------
    text : str
        The raw article title or body text.
    remove_leakage : bool
        Whether to strip known source signatures (e.g. Reuters datelines, image credits).
    lemmatize : bool
        Whether to reduce tokens to dictionary base forms using WordNetLemmatizer.
    remove_stopwords : bool
        Whether to eliminate common English stopwords.

    Returns:
    -------
    str : Cleaned, normalized single-space string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    raw = text.strip()

    # 1. Strip data leakage datelines (e.g. "WASHINGTON (Reuters) - ") at the start of articles
    if remove_leakage:
        raw = DATELINE_PATTERN.sub("", raw)
        raw = REUTERS_PATTERN.sub("", raw)
        raw = WATERMARKS_PATTERN.sub("", raw)
        raw = TWITTER_PIC_PATTERN.sub("", raw)

    # 2. Lowercase
    cleaned = raw.lower()

    # 3. Strip URLs, Twitter handles, and HTML
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = TWITTER_USER_PATTERN.sub(" ", cleaned)
    cleaned = HTML_PATTERN.sub(" ", cleaned)

    # 4. Remove punctuation
    cleaned = cleaned.translate(PUNCT_TRANSLATOR)

    # 5. Remove numbers
    cleaned = NUMBERS_PATTERN.sub(" ", cleaned)

    # 6. Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        # Fallback to whitespace/regex splitting if NLTK tokenizer encounters an issue
        tokens = cleaned.split()

    # 7. Filter stopwords, small noise tokens, and lemmatize
    processed_tokens: list[str] = []
    for token in tokens:
        token = token.strip()
        if len(token) <= 1:
            continue
        if remove_stopwords and token in STOP_WORDS:
            continue
        if lemmatize and LEMMATIZER is not None:
            token = LEMMATIZER.lemmatize(token)
        processed_tokens.append(token)

    # 8. Reassemble
    result = " ".join(processed_tokens)
    return WHITESPACE_PATTERN.sub(" ", result).strip()


def get_cleaned_dataset(
    data_dir: Path | str = DATA_DIR,
    force_recompute: bool = False,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve the cleaned dataset. If cached to CSV, load directly from disk.
    Otherwise, load raw True.csv and Fake.csv, clean them, and save the cache.

    Parameters:
    ----------
    data_dir : Path | str
        Path to the data directory containing True.csv and Fake.csv.
    force_recompute : bool
        If True, ignore existing cached CSV and re-process from raw data.
    sample_size : Optional[int]
        If specified, sample down to N rows for fast debugging/testing.

    Returns:
    -------
    pd.DataFrame : Cleaned DataFrame containing columns ['title', 'text', 'full_text', 'cleaned_text', 'label'].
    """
    ensure_dirs()
    cache_path = Path(data_dir) / "cleaned_news.csv"

    if cache_path.exists() and not force_recompute:
        df = pd.read_csv(cache_path)
        # Handle possible NaN values in cleaned_text
        df["cleaned_text"] = df["cleaned_text"].fillna("")
        df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return df

    # Load raw dataset (True + Fake, label leakage columns removed, deduplicated)
    df_raw = load_raw_dataset(data_dir)

    # Clean the full combined text
    df_raw["cleaned_text"] = df_raw["full_text"].apply(
        lambda t: clean_text(t, remove_leakage=True, lemmatize=True, remove_stopwords=True)
    )

    # Drop any rows that became completely empty after cleaning
    df_cleaned = df_raw[df_raw["cleaned_text"].str.strip() != ""].copy()
    df_cleaned = df_cleaned.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)

    # Cache to CSV
    df_cleaned.to_csv(cache_path, index=False)

    if sample_size and len(df_cleaned) > sample_size:
        df_cleaned = df_cleaned.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df_cleaned


if __name__ == "__main__":
    print("Executing preprocessing pipeline test...")
    df = get_cleaned_dataset()
    print(f"Dataset successfully prepared. Total articles: {len(df)}")
    print(f"Class distribution: Real (0): {(df['label'] == 0).sum()}, Fake (1): {(df['label'] == 1).sum()}")
    print("Sample cleaned text snippet:")
    print(df["cleaned_text"].iloc[0][:200])
