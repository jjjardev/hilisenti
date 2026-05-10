"""
HiliSenti Training Script
Train XLM‑RoBERTa‑large for Hiligaynon sentiment analysis on the
jjjardev/hilisenti‑v1 dataset.
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
DATASET_NAME = "jjjardev/hilisenti-v1"  # Hugging Face dataset
OUTPUT_DIR = "./hilisenti_model"        # where to save model & tokenizer
MAX_LENGTH = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
GRADIENT_ACCUMULATION = 2
SEED = 42

# ---------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------
def normalize_hiligaynon(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKC', str(text))
    text = text.lower()
    text = re.sub(r'\b(ha){2,}[h]*\b', 'hahaha', text)
    text = re.sub(r'\b(he){2,}[h]*\b', 'hehehe', text)
    text = re.sub(r'\b([a-z]{3,})2\b', r'\1-\1', text)
    text = re.sub(r'(\w+)-\1', r'\1 \1', text)
    replacements = {
        r'\bwla\b': 'wala', r'\bwaay\b': 'wala', r'\bway\b': 'wala',
        r'\bndi\b': 'indi', r'\bnd\b': 'indi',
        r'\bgd\b': 'gid', r'\bgud\b': 'gid', r'\bmn\b': 'man',
        r'\bnmn\b': 'naman', r'\bna lng\b': 'nalang', r'\bnlng\b': 'nalang',
        r'\bbl\b': 'bala', r'\btni\b': 'tani', r'\btne\b': 'tani',
        r'\bsya\b': 'siya', r'\bxa\b': 'siya', r'\bxia\b': 'siya',
        r'\bnya\b': 'niya', r'\bnyo\b': 'ninyo', r'\bcmu\b': 'sa imo',
        r'\bsakn\b': 'sa akon', r'\bskn\b': 'sa akon', r'\bkw\b': 'ikaw',
        r'\bky\b': 'kay', r'\bmng\b': 'mga', r'\bdpt\b': 'dapat',
        r'\bsbng\b': 'subong', r'\bkrn\b': 'karon', r'\bhlng\b': 'halong',
        r'\bamo\b': 'amo', r'\bamu\b': 'amo',
        r'\bpro\b': 'pero', r'\bpru\b': 'pero',
        r'\bkg\b': 'kag', r'\bkng\b': 'kon', r'\bkun\b': 'kon',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------
# METRICS
# ---------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1_per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2])
    precision_per_class = precision_score(labels, predictions, average=None, labels=[0, 1, 2])
    recall_per_class = recall_score(labels, predictions, average=None, labels=[0, 1, 2])
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "f1_negative": f1_per_class[0],
        "f1_neutral": f1_per_class[1],
        "f1_positive": f1_per_class[2],
        "precision_negative": precision_per_class[0],
        "precision_neutral": precision_per_class[1],
        "precision_positive": precision_per_class[2],
        "recall_negative": recall_per_class[0],
        "recall_neutral": recall_per_class[1],
        "recall_positive": recall_per_class[2],
    }

def main():
    # Load dataset from Hugging Face
    dataset = load_dataset(DATASET_NAME)
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Clean labels (should already be integers)
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].astype(int)

    # Normalize text
    print("Normalizing datasets...")
    for df in [train_df, val_df, test_df]:
        df["sentence"] = df["sentence"].apply(normalize_hiligaynon)

    print(f"Train: {len(train_df)} samples, Val: {len(val_df)}, Test: {len(test_df)}")

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("Class Weights:", class_weights_tensor)

    # Tokenizer & model
    model_ckpt = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )

    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_pandas(train_df[['sentence', 'label']])
    val_ds = HFDataset.from_pandas(val_df[['sentence', 'label']])
    test_ds = HFDataset.from_pandas(test_df[['sentence', 'label']])

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    tokenized_test = test_ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    )

    # Custom trainer with weighted loss
    class CustomTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=0.05,
        warmup_ratio=0.10,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-6},
        optim="adamw_torch_fused",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",  # set to "tensorboard" if you have it
        save_total_limit=2,
        seed=SEED,
    )

    trainer = CustomTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    # Final evaluation on test set
    test_results = trainer.evaluate(tokenized_test)
    print(f"\nTest Results: {test_results}")

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
