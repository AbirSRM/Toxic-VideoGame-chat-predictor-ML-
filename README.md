# Toxic Video Game Chat Prediction using DistilBERT

A multi-label NLP classifier that detects five categories of toxic language in video game chat using a fine-tuned DistilBERT transformer model. This repository includes both a Google Colab notebook for experimentation and a production-ready Python backend with a REST API and Docker deployment.

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

## Python Backend & API

The repository has been expanded to include a modular Python backend for training, evaluation, and deployment via FastAPI and Docker.

### Project Structure

```
├── api.py                   # FastAPI REST endpoint for live predictions
├── config.py                # Hyperparameters and paths configuration
├── Dockerfile               # Docker configuration for deployment
├── docker-compose.yml       # Docker Compose setup
├── evaluate.py              # Script for evaluating model performance
├── infer.py                 # Command-line inference script
├── model.py                 # PyTorch model definitions
├── requirements.txt         # Python dependencies
├── train.py                 # Main training script
├── train_both.py            # Script for combined training
├── test_*.py                # Unit tests
└── Toxic_Videogame_Chat_Prediction_using_DistilBERT_(1).ipynb # Original Colab notebook
```

### Running the API (Docker)

You can easily run the inference API using Docker Compose:

```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`. You can test it by sending a POST request to the `/predict` endpoint.

### Training Locally

To train the model locally using the Python scripts:
```bash
python train.py
```

To evaluate the trained model:
```bash
python evaluate.py
```

To run inference from the command line:
```bash
python infer.py
```

---

## Google Colab Experimentation (Legacy)

For those looking to explore the original experimental setup or run the training pipeline in a hosted GPU environment, the original Google Colab notebook is retained in this repository.

### How to Run in Colab

1. Open `Toxic_Videogame_Chat_Prediction_using_DistilBERT_(1).ipynb` in Google Colab.
2. Upload `train.csv` to your Google Drive.
3. Update `file_path` in Cell 2 to match your Drive path:
   ```python
   file_path = '/content/drive/MyDrive/YOUR_FOLDER/train.csv'
   ```
4. Run all cells in order (Colab GPU runtime recommended).
5. Trained model checkpoints are saved to `./results/`.

---

## Key Design Decisions

**Weighted loss function:** A custom `WeightedTrainer` computes per-class `pos_weight` from the training partition only (neg_count / pos_count per label), then passes it to `BCEWithLogitsLoss`. This directly penalises the model more for missing rare toxic labels.

**Oversampling:** All toxic-flagged rows are duplicated 2x in the training set before tokenisation, giving the model more exposure to minority-class patterns without augmenting the validation set.

**Early stopping:** `EarlyStoppingCallback(patience=1)` halts training if micro-F1 on the validation set does not improve, preventing overfitting on the small sample.

**Stratified split:** The train/val split stratifies on a derived `toxic_binary` column (any label active = 1), ensuring proportional toxic representation in both partitions.

---

## Tech Stack

| Component | Library |
|---|---|
| Transformer model | `transformers` (HuggingFace) |
| Dataset handling | `datasets` (HuggingFace) |
| Training loop | PyTorch & `Trainer` API |
| API & Deployment | FastAPI, Docker |
| Metrics | `scikit-learn` |
| Notebook Environment | Google Colab (GPU) |

---

## Limitations & Future Work

- Trained on only 50k of the full dataset — scaling to the full corpus would improve recall on rare categories like `threat` and `identity_attack`
- Threshold of 0.5 is fixed; per-label threshold tuning on a held-out set could improve macro-F1
- No cross-game or cross-platform validation; domain shift from Wikipedia comments (Jigsaw source) to actual game chat may affect performance
- Planned upgrade: integrate directly with game chat pipelines via WebSocket.
- Planned: experiment with `roberta-base` and `distilbert-base-multilingual` for non-English game lobbies
