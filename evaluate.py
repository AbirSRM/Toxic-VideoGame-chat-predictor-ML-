import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from config import Config


def evaluate(tier: str):
    cfg        = Config.get_tier(tier)
    model_path = f"./models/{tier}_tier/final_model"
    device     = Config.DEVICE

    print(f"[+] Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # ── Load & merge test set ─────────────────────────────────────────────────
    # test.csv           → comment_text (no labels)
    # test_labels.csv    → toxic {0, 1, -1}  (-1 = not scored by Kaggle)
    # Result after merge → ~63,978 rows with real binary labels
    print("[+] Loading and merging test data...")
    test_text   = pd.read_csv(Config.TEST_PATH)
    test_labels = pd.read_csv(Config.TEST_LABELS_PATH,
                              usecols=["id", Config.TEST_LABEL_COLUMN])

    merged = test_text.merge(test_labels, on="id")
    merged = merged[merged[Config.TEST_LABEL_COLUMN] != -1].reset_index(drop=True)
    merged[Config.TEXT_COLUMN] = merged[Config.TEXT_COLUMN].fillna("").astype(str)

    print(f"[+] Usable test samples : {len(merged):,}")
    print(f"[+] Toxic in test set   : {merged[Config.TEST_LABEL_COLUMN].sum():,} "
          f"({merged[Config.TEST_LABEL_COLUMN].mean()*100:.1f}%)")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    def tokenize(examples):
        return tokenizer(examples[Config.TEXT_COLUMN],
                         truncation=True, max_length=cfg.max_len)

    dataset   = HFDataset.from_pandas(merged[[Config.TEXT_COLUMN]])
    tokenized = dataset.map(tokenize, batched=True, remove_columns=[Config.TEXT_COLUMN])
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(tokenized, batch_size=64, collate_fn=collator)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_probs, all_preds = [], []
    total_batches = len(loader)
    print("[+] Running inference...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs   = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds   = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            if (i + 1) % 50 == 0:
                print(f"    Batch {i+1}/{total_batches}")

    true_labels = merged[Config.TEST_LABEL_COLUMN].values

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'-'*55}")
    print(f"  Evaluation — {tier.upper()} Tier ({cfg.model_name})")
    print(f"{'-'*55}")
    print(classification_report(true_labels, all_preds,
                                 target_names=["Non-Toxic", "Toxic"], digits=4))
    auc = roc_auc_score(true_labels, all_probs)
    print(f"AUC-ROC : {auc:.4f}")
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, all_preds)
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on Jigsaw test set")
    parser.add_argument("--tier", type=str, choices=["high", "distil_high", "low"], default="low")
    args = parser.parse_args()
    evaluate(args.tier)
