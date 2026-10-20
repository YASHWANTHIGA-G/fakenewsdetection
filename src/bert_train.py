"""
Verdict - DistilBERT Deep Learning Benchmark Script (Optional Comparison).

Provides end-to-end PyTorch + Hugging Face Transformers fine-tuning on the news dataset
for deep contextual representation benchmarking against classical TF-IDF models.

Usage:
    python -m src.bert_train [--epochs 3] [--batch-size 16] [--sample 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.preprocess import get_cleaned_dataset
from src.utils import MODELS_DIR, ensure_dirs


def run_bert_training(
    epochs: int = 3,
    batch_size: int = 16,
    sample_size: int | None = 2000,
    lr: float = 2e-5,
) -> None:
    """Fine-tune DistilBERT on news dataset."""
    ensure_dirs()
    print("=" * 60)
    print("DISTILBERT FINE-TUNING BENCHMARK")
    print("=" * 60)

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AdamW,
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        print(
            "DistilBERT dependencies not installed. To run transformer benchmarking, "
            "install PyTorch and Hugging Face Transformers:\n"
            "  pip install torch transformers\n"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computational device: {device}")

    # Load dataset
    df = get_cleaned_dataset(sample_size=sample_size)
    print(f"Loaded {len(df)} samples for transformer training.")

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["cleaned_text"].values,
        df["label"].values,
        test_size=0.20,
        random_state=42,
        stratify=df["label"].values,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    ).to(device)

    class NewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len: int = 256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = int(self.labels[idx])
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

    train_loader = DataLoader(
        NewsDataset(X_train_text, y_train, tokenizer),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        NewsDataset(X_val_text, y_val, tokenizer),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    print(f"Training DistilBERT across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                trues.extend(labels.cpu().numpy())

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, zero_division=0)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

    bert_save_path = MODELS_DIR / "distilbert_finetuned"
    bert_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bert_save_path)
    tokenizer.save_pretrained(bert_save_path)
    print(f"DistilBERT fine-tuned model weights saved to: {bert_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for fake news classification.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--sample", type=int, default=2000, help="Subset size for training")
    args = parser.parse_args()

    run_bert_training(epochs=args.epochs, batch_size=args.batch_size, sample_size=args.sample)
