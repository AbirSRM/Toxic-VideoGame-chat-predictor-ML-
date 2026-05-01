# Toxic Video Game Chat Prediction using DistilBERT

A multi-label NLP classifier that detects five categories of toxic language in video game chat using a fine-tuned DistilBERT transformer model.

---

## Problem Statement

Toxic behaviour in online gaming — insults, threats, identity attacks — degrades player experience and drives people away from games. Automated moderation systems need to be nuanced enough to distinguish harmless banter ("GG noob!") from genuinely harmful language. This project builds a multi-label classifier that flags five distinct toxicity types simultaneously per comment.

---

## Dataset

**Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip) (Kaggle)

- 50,000 rows sampled from the full training set
- Each row: a `comment_text` string + 6 binary label columns
- Labels used in this project (5 of 6):

| Label | Description |
|---|---|
| `severe_toxicity` | Extremely harmful or hateful language |
| `obscene` | Profane or vulgar content |
| `identity_attack` | Attacks based on identity (race, gender, etc.) |
| `insult` | Direct personal insults |
| `threat` | Explicit threats of harm |

**Class imbalance:** Toxic comments are a minority class across all labels, addressed via oversampling (2x) of toxic samples during training.

---

## Model Architecture

- **Base model:** `distilbert-base-uncased` (HuggingFace Transformers)
- **Task type:** Multi-label sequence classification
- **Output:** 5 sigmoid-activated probabilities, one per toxicity category
- **Decision threshold:** 0.5 per label

---

## Pipeline

```
Raw CSV
  └─► Sample 50k rows
        └─► Threshold float labels at 0.5 → binary
              └─► Stratified train/val split (80/20)
                    └─► Oversample toxic minority class (2x)
                          └─► Tokenize with DistilBERT (max_length=256)
                                └─► Fine-tune with weighted BCEWithLogitsLoss
                                      └─► Evaluate with micro/macro F1
```

---

## Key Design Decisions

**Weighted loss function:** A custom `WeightedTrainer` computes per-class `pos_weight` from the training partition only (neg_count / pos_count per label), then passes it to `BCEWithLogitsLoss`. This directly penalises the model more for missing rare toxic labels.

**Oversampling:** All toxic-flagged rows are duplicated 2x in the training set before tokenisation, giving the model more exposure to minority-class patterns without augmenting the validation set.

**Early stopping:** `EarlyStoppingCallback(patience=1)` halts training if micro-F1 on the validation set does not improve, preventing overfitting on the small sample.

**Stratified split:** The train/val split stratifies on a derived `toxic_binary` column (any label active = 1), ensuring proportional toxic representation in both partitions.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall exact-match accuracy |
| Precision (micro) | Pooled precision across all labels |
| Recall (micro) | Pooled recall across all labels |
| **F1-score (micro)** | **Primary training metric** |
| F1-score (macro) | Unweighted average F1 across labels |

Per-class confusion matrices (TP, FP, FN, TN) are printed for all 5 labels after evaluation.

---

## Example Predictions

```python
predict_toxicity("You are an amazing player! Great game!")
# → all scores near 0.0 (clean)

predict_toxicity("I hate you, you stupid noob!")
# → insult, obscene scores elevated

predict_toxicity("You suck at this game, uninstall!")
# → insult score elevated

predict_toxicity("This team is trash, everyone is terrible")
# → insult, obscene scores elevated
```

---

## Tech Stack

| Component | Library |
|---|---|
| Transformer model | `transformers` (HuggingFace) |
| Dataset handling | `datasets` (HuggingFace) |
| Training loop | `Trainer` API |
| Metrics | `scikit-learn` |
| Runtime | Google Colab (GPU) |

---

## Limitations & Future Work

- Trained on only 50k of the full dataset — scaling to the full corpus would improve recall on rare categories like `threat` and `identity_attack`
- Threshold of 0.5 is fixed; per-label threshold tuning on a held-out set could improve macro-F1
- No cross-game or cross-platform validation; domain shift from Wikipedia comments (Jigsaw source) to actual game chat may affect performance
- Planned upgrade: deploy as a REST API endpoint for real-time inference in game chat pipelines
- Planned: experiment with `roberta-base` and `distilbert-base-multilingual` for non-English game lobbies

---

## How to Run

1. Upload `train.csv` to Google Drive
2. Update `file_path` in Cell 2 to match your Drive path:
   ```python
   file_path = '/content/drive/MyDrive/YOUR_FOLDER/train.csv'
   ```
3. Run all cells in order (Colab GPU runtime recommended)
4. Trained model checkpoints are saved to `./results/`

---

## Repository Structure

```
├── Toxic_Videogame_Chat_Prediction_using_DistilBERT.ipynb
├── README.md
└── requirements.txt
```

## Requirements

```
transformers
datasets
accelerate
scikit-learn
torch
pandas
numpy
```
