"""
HiliSenti Training Script
Train XLM-RoBERTa-large for Hiligaynon sentiment analysis on the
jjjardev/hilisenti-v1 dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset as HFDataset
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

from preprocess import HILIGAYNON_TOKENS, normalize_frame

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
DATASET_NAME = "jjjardev/hilisenti-v1"  # Hugging Face dataset
OUTPUT_DIR = "./hilisenti_model"        # where to save model & tokenizer
BASE_MODEL = "xlm-roberta-large"
MAX_LENGTH = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
GRADIENT_ACCUMULATION = 2
# Label smoothing is 0.0 to match the released checkpoint. The saved
# training_args.bin from that run records label_smoothing_factor = 0.1, but
# Trainer's built-in loss -- the only consumer of that setting -- is replaced
# by the class-weighted objective in CustomTrainer, so the factor never
# applied. Set this to 0.1 to train with smoothing; it will not reproduce the
# released weights.
LABEL_SMOOTHING = 0.0
SEED = 42

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ---------------------------------------
# TOKENIZER VOCABULARY AUGMENTATION
# ---------------------------------------
def add_hiligaynon_tokens(tokenizer, model):
    """Add the 46 in-vocabulary Hiligaynon terms and grow the embedding matrix.

    The released checkpoint has vocab_size 250048 against the base model's
    250002, so the terms must be appended in the order given by
    HILIGAYNON_TOKENS to land on the same ids. New rows are initialised to the
    mean of the sub-word embeddings the base tokenizer would have produced,
    which keeps them in-distribution with the pre-trained space.
    """
    added = tokenizer.add_tokens(HILIGAYNON_TOKENS)
    if added != len(HILIGAYNON_TOKENS):
        raise RuntimeError(
            f"expected to add {len(HILIGAYNON_TOKENS)} tokens, added {added}"
        )
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    with torch.no_grad():
        for token in HILIGAYNON_TOKENS:
            pieces = tokenizer.tokenize(token)
            if not pieces:
                raise RuntimeError(f"{token!r} tokenized to nothing")
            subword_ids = tokenizer.convert_tokens_to_ids(pieces)
            token_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings[token_id] = input_embeddings[subword_ids].mean(dim=0)

    return tokenizer, model


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
    train_df = normalize_frame(dataset["train"].to_pandas())
    val_df = normalize_frame(dataset["validation"].to_pandas())
    test_df = normalize_frame(dataset["test"].to_pandas())

    # Clean labels (should already be integers)
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].astype(int)

    print(f"Train: {len(train_df)} samples, Val: {len(val_df)}, Test: {len(test_df)}")

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("Class Weights:", class_weights_tensor)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )

    # Grow the vocabulary before tokenizing so the new terms are single tokens
    tokenizer, model = add_hiligaynon_tokens(tokenizer, model)
    print(f"Vocabulary: {len(tokenizer)} tokens "
          f"(embedding matrix {model.get_input_embeddings().num_embeddings})")

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = HFDataset.from_pandas(train_df[['sentence', 'label']])
    val_ds = HFDataset.from_pandas(val_df[['sentence', 'label']])
    test_ds = HFDataset.from_pandas(test_df[['sentence', 'label']])

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])
    tokenized_test = test_ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])

    class WeightedLoss(nn.Module):
        """Class-weighted cross-entropy, optionally label-smoothed.

        Trainer only applies TrainingArguments.label_smoothing_factor inside
        its own loss, so overriding compute_loss without handling it makes the
        setting a silent no-op. When epsilon is 0 this reduces exactly to
        weighted cross-entropy, which is what produced the released weights.
        Smoothing blends the weighted target distribution with a uniform one.
        """

        def __init__(self, weight, epsilon):
            super().__init__()
            self.weight = weight
            self.epsilon = epsilon

        def forward(self, logits, labels):
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            weighted_nll = -log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)
            if self.weight is not None:
                weighted_nll = weighted_nll * self.weight[labels]
            smooth = -log_probs.mean(dim=-1)
            if self.weight is not None:
                smooth = smooth * self.weight[labels]
            return ((1.0 - self.epsilon) * weighted_nll
                    + self.epsilon * smooth).mean()

    class CustomTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = WeightedLoss(
                weight=self.class_weights.to(model.device),
                epsilon=LABEL_SMOOTHING,
            )
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
        report_to="tensorboard",
        # load_best_model_at_end needs at least two checkpoints retained: the
        # best one and the one it is compared against.
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

    # Per-class report and confusion matrix, matching Table 3 and Figure 2
    predictions = np.argmax(trainer.predict(tokenized_test).predictions, axis=-1)
    gold = test_df["label"].to_numpy()
    print("\nClassification report:")
    print(classification_report(gold, predictions, target_names=list(LABELS.values()),
                                digits=3, zero_division=0))
    cm = confusion_matrix(gold, predictions, labels=[0, 1, 2])
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
