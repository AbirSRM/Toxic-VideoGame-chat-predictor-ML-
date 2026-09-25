import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset

from config import Config
from model import get_model_and_tokenizer


# ── Custom Trainer: Weighted Cross-Entropy for class imbalance ────────────────
class WeightedLossTrainer(Trainer):
    """
    Extends HuggingFace Trainer to apply per-class loss weighting.
    This directly addresses the ~9:1 non-toxic:toxic imbalance in the dataset,
    preventing the model from collapsing to always predicting 'Non-Toxic'.
    """
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        # Move weights to same device as logits at compute time (handles CPU/GPU)
        weights = self.class_weights.to(logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs_toxic  = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    acc = float((predictions == labels).mean())
    f1  = f1_score(labels, predictions, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs_toxic)
    except ValueError:
        auc = 0.0

    return {"accuracy": acc, "f1": f1, "auc_roc": auc}


# ── Main Training Function ────────────────────────────────────────────────────
def run_training(tier: str, sample_size: int = None):
    cfg = Config.get_tier(tier)
    model, tokenizer = get_model_and_tokenizer(tier)

    # Gradient checkpointing trades compute for memory — useful for high-tier on CPU
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print("[+] Gradient checkpointing enabled")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    # Only read columns we need — avoids loading all 45 train.csv columns into RAM
    usecols = ["id", Config.TEXT_COLUMN, Config.LABEL_COLUMN]
    if sample_size:
        print(f"[!] DRY-RUN: Loading first {sample_size} rows...")
        df = pd.read_csv(Config.TRAIN_PATH, nrows=sample_size, usecols=usecols)
    else:
        print(f"[+] Loading full dataset from {Config.TRAIN_PATH}")
        df = pd.read_csv(Config.TRAIN_PATH, usecols=usecols)

    df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].fillna("").astype(str)

    # Jigsaw 2019 uses soft float labels -> binarise at 0.5
    df["labels"] = (df[Config.LABEL_COLUMN] >= 0.5).astype(int)
    df = df[[Config.TEXT_COLUMN, "labels"]]

    # ── 2. Compute class weights to handle imbalance ──────────────────────────
    n_total   = len(df)
    n_toxic   = int(df["labels"].sum())
    n_nontox  = n_total - n_toxic
    w_nontox  = n_total / (2 * n_nontox)
    w_toxic   = n_total / (2 * n_toxic)
    class_weights = torch.tensor([w_nontox, w_toxic], dtype=torch.float)
    print(f"[+] Class distribution -> Non-toxic: {n_nontox:,} | Toxic: {n_toxic:,}")
    print(f"[+] Loss weights       -> Non-toxic: {w_nontox:.4f} | Toxic: {w_toxic:.4f}")

    # ── 3. Stratified Train / Val Split ──────────────────────────────────────
    # Stratify preserves class ratio in both splits
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["labels"]
    )

    train_hf = HFDataset.from_pandas(train_df.reset_index(drop=True))
    val_hf   = HFDataset.from_pandas(val_df.reset_index(drop=True))

    # ── 4. Tokenize (no static padding — DataCollator handles it per-batch) ───
    def tokenize(examples):
        return tokenizer(
            examples[Config.TEXT_COLUMN],
            truncation=True,
            max_length=cfg.max_len,
            # No padding="max_length" here — dynamic padding is more efficient
        )

    print("[+] Tokenizing dataset...")
    remove_cols = [c for c in train_hf.column_names if c != "labels"]
    tokenized_train = train_hf.map(tokenize, batched=True, remove_columns=remove_cols)
    tokenized_val   = val_hf.map(tokenize,   batched=True, remove_columns=remove_cols)

    # ── 5. Training Arguments ─────────────────────────────────────────────────
    num_epochs = 1 if sample_size else cfg.epochs

    # Calculate effective batch size and warmup
    eff_batch_size = cfg.batch_size
    grad_accum_steps = max(1, 32 // eff_batch_size) # Aim for effective batch size of 32
    total_steps = (len(tokenized_train) // (eff_batch_size * grad_accum_steps)) * num_epochs
    calculated_warmup_steps = max(500, int(total_steps * 0.1))

    training_args = TrainingArguments(
        output_dir                  = Config.output_dir(tier),
        eval_strategy               = "steps",
        eval_steps                  = 5000 if not sample_size else 10,
        save_strategy               = "steps",
        save_steps                  = 5000 if not sample_size else 10,
        learning_rate               = cfg.lr,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        gradient_accumulation_steps = grad_accum_steps,
        num_train_epochs            = num_epochs,
        weight_decay                = 0.01,
        fp16                        = cfg.use_fp16,
        logging_steps               = 10 if sample_size else 100,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",        # F1 > accuracy for imbalanced data
        greater_is_better           = True,
        save_total_limit            = 2,
        report_to                   = "none",
        dataloader_num_workers      = 0,           # 0 = safe across all OS configs
        warmup_steps                = calculated_warmup_steps,
    )

    from transformers import EarlyStoppingCallback

    trainer = WeightedLossTrainer(
        class_weights    = class_weights,
        model            = model,
        args             = training_args,
        train_dataset    = tokenized_train,
        eval_dataset     = tokenized_val,
        processing_class = tokenizer,
        data_collator    = DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"[+] Starting | Tier: {tier.upper()} | Device: {Config.DEVICE.upper()} | FP16: {cfg.use_fp16} | GradAccum: {grad_accum_steps}")
    trainer.train()

    # ── 6. Save final model ───────────────────────────────────────────────────
    save_path = os.path.join(Config.output_dir(tier), "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[✓] Done. Model saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ONLYPosiChat toxicity classifier")
    parser.add_argument("--tier",   type=str, choices=["high", "distil_high", "low"], default="low",
                        help="'high' for RoBERTa | 'distil_high' for DistilRoBERTa w/ high params | 'low' for DistilRoBERTa")
    parser.add_argument("--sample", type=int, default=None,
                        help="Rows to load for a quick dry-run (e.g. 2000)")
    args = parser.parse_args()
    run_training(args.tier, args.sample)