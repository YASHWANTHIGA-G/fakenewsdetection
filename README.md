# Verdict — Production-Quality Fake News Detection System

Verdict is a production-grade machine learning system and web application designed to evaluate the rhetorical style, sensationalism, and lexical credibility of news articles. It combines classical NLP, statistical machine learning with 3-fold cross-validation grid search, linear explainability signals, Optical Character Recognition (OCR), document parsing, and an editorial "fact-checking desk" user interface.

---

## Table of Contents
1. [Architectural Overview](#architectural-overview)
2. [Project Structure](#project-structure)
3. [Installation & Setup](#installation--setup)
4. [Dataset & Data Preparation](#dataset--data-preparation)
5. [Pipeline Execution Guide](#pipeline-execution-guide)
   - [Step 1: Exploratory Data Analysis (EDA)](#step-1-exploratory-data-analysis-eda)
   - [Step 2: Model Training & Hyperparameter Tuning](#step-2-model-training--hyperparameter-tuning)
   - [Step 3: Launching the Web Application](#step-3-launching-the-web-application)
   - [Step 4: Optional Transformer Baseline (DistilBERT)](#step-4-optional-transformer-baseline-distilbert)
6. [Multi-Input Support: Text, Image (OCR), Document](#multi-input-support-text-image-ocr-document)
7. [Explainability & Lexical Signals](#explainability--lexical-signals)
8. [Expected Results & Benchmark Metrics](#expected-results--benchmark-metrics)
9. [Known Limitations & Biases](#known-limitations--biases)
10. [Future Roadmap](#future-roadmap)

---

## Architectural Overview

Verdict approaches misinformation detection from a **stylometric and lexical framing** perspective:
- **Preprocessing & Anti-Leakage Filter:** Strips publisher datelines (e.g. `WASHINGTON (Reuters) -`), watermarks (`getty images`, `21st century wire`), Twitter image handles (`pic.twitter.com`), URLs, HTML tags, digits, and punctuation. Lemmatizes terms with WordNet.
- **Feature Extraction:** Sublinear TF-IDF vectorization with unigrams and bigrams ($1 \le n \le 2$), capturing up to 50,000 discriminative n-grams.
- **GridSearchCV Model Selection:** Evaluates Logistic Regression, Multinomial Naive Bayes, Passive Aggressive Classifier, Linear SVM, and Random Forest using 3-fold cross-validation optimizing for the **F1-Score**.
- **Probability Calibration:** Decision function margins for margin classifiers (SVM, Passive Aggressive) are dynamically calibrated into continuous probabilities via numerically stable sigmoid scaling.
- **Local Feature Attribution:** Computes individual token contributions ($\text{contribution} = \text{TF-IDF} \times \beta_{\text{LogisticRegression}}$) to highlight exactly which words pushed the verdict toward "Fake" or "Real".
- **Multi-Modal Ingestion:** Ingests raw text, document files (`.pdf`, `.txt`), or image clippings (`.png`, `.jpg`, `.jpeg`, `.webp`) via in-memory EasyOCR and `pypdf`.

---

## Project Structure

```
aimlproject/
├── data/
│   ├── True.csv              # Real news articles (Reuters crawl)
│   ├── Fake.csv              # Fabricated / sensationalist articles
│   └── cleaned_news.csv      # Cached preprocessed and sanitized dataset
├── models/
│   ├── model.joblib          # Best performing tuned classifier (selected by CV F1)
│   ├── vectorizer.joblib     # Fitted TF-IDF vectorizer (50k vocabulary)
│   ├── explainer.joblib      # Tuned Logistic Regression used for token signals
│   └── metrics.json          # Performance metrics, evaluation table, and top words
├── reports/
│   ├── class_distribution.png       # Real vs Fake article count
│   ├── text_length_histogram.png    # Word-length distribution by class
│   ├── wordcloud_real.png           # Lexical cloud for authentic reports
│   ├── wordcloud_fake.png           # Lexical cloud for fabricated stories
│   ├── top20_words.png              # Top-20 term frequencies per class
│   ├── model_comparison.csv         # Comparative evaluation table (Acc, F1, AUC, etc.)
│   ├── model_comparison.png         # Grouped performance bar chart
│   ├── confusion_matrices.png       # Subplot grid of test confusion matrices
│   ├── roc_curves.png               # Combined ROC curves for all models
│   └── top_features.png             # Top 20 Real and Fake indicator words
├── src/
│   ├── __init__.py           # Package initializer
│   ├── utils.py              # Path definitions, sigmoid calibration, PDF/TXT loaders
│   ├── preprocess.py         # NLTK downloader, leakage filters, clean_text()
│   ├── ocr.py                # Image contrast enhancement and EasyOCR lazy loader
│   ├── eda.py                # Exploratory charts generation
│   ├── train.py              # GridSearch, evaluation, and artifact saving
│   ├── predict.py            # Predict function with signals and confidence rules
│   └── bert_train.py         # Optional DistilBERT transformer fine-tuning baseline
├── templates/
│   └── index.html            # Fact-Checking Desk UI
├── static/
│   ├── style.css             # Newsreader & Public Sans desk design system
│   └── app.js                # Tab switching, drag/drop, paste, safe DOM rendering
├── app.py                    # Flask REST API and web application server
├── requirements.txt          # Pinned project dependencies
└── README.md                 # Full documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Virtual environment tool (`venv`)

### 1. Clone or Open the Workspace
```bash
cd aimlproject
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note on EasyOCR & PyTorch:** When first installed, EasyOCR bundles PyTorch and OpenCV. On its first image OCR run, EasyOCR will automatically download approximately 100 MB of English CRAFT text-detection and recognition models into your user cache. No system-level external binary installations are required.

---

## Dataset & Data Preparation

This project uses the **ISOT Fake and Real News Dataset** compiled by the Information Security and Object Technology (ISOT) Research Lab:
- **Canonical Kaggle Source:** [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files Required in `data/`:**
  - `data/True.csv` (contains ~21,417 articles from Reuters)
  - `data/Fake.csv` (contains ~23,481 articles flagged by PolitiFact/Wikipedia)

### Schema:
Both files contain `title`, `text`, `subject`, and `date`.
- **Target Label:** `0` = Real, `1` = Fake.
- **Anti-Leakage Guarantee:** The columns `subject` (e.g. `politicsNews` vs `News`) and `date` are **strictly dropped**, as they directly reveal the dataset partition.
- **Starter Mode:** If you run the scripts before downloading the 100 MB Kaggle archive, Verdict will automatically initialize a representative benchmark starter dataset so you can run the entire pipeline end-to-end immediately!

---

## Pipeline Execution Guide

Run each module from the project root using `python -m <module>`:

### Step 1: Exploratory Data Analysis (EDA)
Inspect class distribution, token lengths, word clouds, and comparative n-gram frequencies:
```bash
python -m src.eda
```
*Generated reports saved to `reports/`:*
- `class_distribution.png`
- `text_length_histogram.png`
- `wordcloud_real.png` & `wordcloud_fake.png`
- `top20_words.png`

### Step 2: Model Training & Hyperparameter Tuning
Train all 5 candidate models using stratified 80/20 train/test splits and 3-fold cross-validation:
```bash
python -m src.train
```

#### Fast Prototyping Flags:
```bash
# Train on a random sample of 2,000 articles for rapid experimentation:
python -m src.train --sample 2000

# Skip Random Forest (which has the longest fit time):
python -m src.train --skip-rf
```

*Outputs produced in `reports/` and `models/`:*
- `models/model.joblib`: The winning estimator (selected by highest Cross-Validation F1).
- `models/vectorizer.joblib`: Fitted 50k TF-IDF vocabulary.
- `models/explainer.joblib`: Tuned Logistic Regression model used for local token attribution.
- `models/metrics.json`: JSON summary containing all test metrics, best model name, and top indicator words.
- `reports/model_comparison.csv` & `reports/model_comparison.png`
- `reports/confusion_matrices.png`
- `reports/roc_curves.png`
- `reports/top_features.png`

### Step 3: Launching the Web Application
Start the Flask backend and editorial Fact-Checking Desk UI:
```bash
python app.py
```
Open your browser to:
```
http://127.0.0.1:5000
```

### Step 4: Optional Transformer Baseline (DistilBERT)
For deep learning practitioners, `src/bert_train.py` provides a PyTorch / Hugging Face transfer learning comparison:
```bash
python -m src.bert_train --epochs 3 --batch-size 16 --sample 2000
```

---

## Multi-Input Support: Text, Image (OCR), Document

Verdict allows users to submit an article in three distinct ways:

### A) Paste Text
- Direct text entry with live word and character counters.
- Built-in one-click samples:
  - **Sober report:** Neutral, factual legislative journalism.
  - **Sensational claim:** Conspiratorial, clickbait phrasing.

### B) Upload Image (Screenshot / News Clipping / Meme)
- Drag-and-drop or click to browse.
- **Clipboard Paste:** Press `Ctrl+V` anywhere on the screen with an image on your clipboard to instantly inspect a captured screenshot.
- Image preprocessing: Proportional upscale (if width < 1000px), grayscale conversion, dynamic range stretching, and contrast enhancement.
- In-memory OCR using EasyOCR (English default; Tesseract/pytesseract noted as alternative).
- If OCR detects fewer than 15 words, a friendly error asks for a clearer image.
- Review and edit OCR output in the collapsible **"Text we read from your upload"** drawer and re-run with one click!

### C) Upload Document (.TXT or .PDF)
- Drag and drop or browse for `.txt` or `.pdf` files.
- Up to 10 pages parsed in memory using `pypdf`.
- Files are parsed directly from memory buffers and are never written to disk.

---

## Explainability & Lexical Signals

Unlike black-box classifiers, Verdict provides full visibility into why a decision was reached:

$$\text{Contribution}(w) = \text{TF-IDF}(w) \times \beta_{\text{LogisticRegression}}(w)$$

- $\text{Contribution} > 0$: The word pushed the model toward **Fake** (e.g. `breaking`, `bombshell`, `unbelievable`, `cabal`, `shocking`).
- $\text{Contribution} < 0$: The word pushed the model toward **Real** (e.g. `spokesman`, `department`, `legislation`, `reuters`, `officials`, `committee`).
- Signals are displayed as normalized color-coded progress bars ($0.0 \text{ to } 1.0$), showing users the most impactful terms.

### Verdict Stamp Rule
- **REAL:** Confidence $\ge 65\%$ toward Real.
- **FAKE:** Confidence $\ge 65\%$ toward Fake.
- **UNCLEAR:** Confidence $< 65\%$. If an article mixes sober policy terms with sensational adjectives, Verdict transparently flags the ambiguity.

---

## Expected Results & Benchmark Metrics

On the full ISOT dataset (~44,800 articles), classical models achieve strong performance on lexical separation:

| Model Architecture | 3-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Passive Aggressive** | ~0.994 | ~99.4% | ~0.994 | ~0.994 | **0.994** | ~0.998 |
| **Linear SVM** | ~0.993 | ~99.3% | ~0.993 | ~0.993 | ~0.993 | ~0.998 |
| **Logistic Regression** | ~0.988 | ~98.8% | ~0.987 | ~0.989 | ~0.988 | ~0.997 |
| **Multinomial Naive Bayes** | ~0.945 | ~94.6% | ~0.942 | ~0.951 | ~0.946 | ~0.984 |
| **Random Forest** | ~0.989 | ~98.9% | ~0.988 | ~0.990 | ~0.989 | ~0.996 |

*(Note: Exact values depend on whether data leakage removal is applied. Without dateline stripping, models achieve artificial 99.9% scores by memorizing "Reuters". With strict leakage removal, models learn genuine lexical styles).*

---

## Known Limitations & Biases

1. **Temporal & Topic Bias:** The dataset reflects 2016–2017 US political and international coverage. Lexical patterns may not generalize accurately to science reporting, celebrity gossip, or events after 2020.
2. **Style vs. Truth:** The model evaluates *stylistic patterns* (sensationalist adjectives, capitalization, rhetorical urgency). A falsehood written in pristine, sober AP/Reuters style will not be caught solely by a lexical classifier.
3. **Language Support:** English only. Articles in Spanish, French, or other languages will not produce valid predictions.
4. **OCR Vulnerabilities:** Blurry screenshots, low-contrast text on noisy backgrounds, or complex multi-column typography may degrade OCR accuracy.
5. **Image Manipulation:** The system reads the *words* inside an image; it does not perform forensic image analysis or detect Photoshop/Deepfake tampering.

---

## Future Roadmap

- [ ] **Live Fact Verification:** Augment lexical predictions with Google Fact Check Tools API and Wikipedia Knowledge Graph retrieval.
- [ ] **Multilingual NLP:** Train XLM-RoBERTa for cross-lingual credibility scoring.
- [ ] **URL Input & Live Web Scraper:** Ingest articles directly from web links with paywall handling and metadata extraction.
- [ ] **Image Forensics & Reverse Image Search:** Integrate reverse search to verify if image clippings are out-of-context or altered.
