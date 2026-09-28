# ONLYPosiChat 🛡️

A modern, auto-scaling hardware-aware chat toxicity classification REST API. Built with FastAPI and HuggingFace Transformers, ONLYPosiChat dynamically selects between optimized model tiers based on the available hardware (CPU vs GPU) to deliver fast and accurate predictions.

---

## 🌟 Key Features

- **Hardware-Aware Auto-Scaling:** Automatically detects hardware capabilities and selects the most efficient model tier:
  - **High Tier (GPU):** Uses `roberta-base` with FP16 precision for maximum accuracy and throughput.
  - **Low Tier (CPU):** Uses `distilroberta-base` for fast, lightweight inference on servers without GPU acceleration.
- **RESTful API:** Clean and documented endpoints powered by FastAPI.
- **Binary Toxicity Classification:** Provides `Toxic` vs `Non-Toxic` classification with continuous probability scores.
- **Configurable Threshold:** Adjust the toxicity probability threshold dynamically per request.
- **Docker Ready:** Effortless deployment using Docker and Docker Compose with model-volume mounting.

---

## 📁 Project Structure

```
├── api.py                   # FastAPI REST endpoint & auto-scaling logic
├── config.py                # Hardware tiers and hyperparameters configuration
├── Dockerfile               # Docker configuration for API deployment
├── docker-compose.yml       # Docker Compose setup for easy orchestration
├── evaluate.py              # Model evaluation scripts
├── infer.py                 # Command-line inference and model loading
├── model.py                 # PyTorch model architecture definitions
├── requirements.txt         # Project dependencies
├── train.py                 # Training script for single tiers
├── train_both.py            # Automated training for both high and low tiers
└── test_*.py                # Unit tests suite
```

---

## 🚀 Getting Started

### 1. Running the API with Docker (Recommended)

The easiest way to get the API running is via Docker Compose.

```bash
# Build and start the container
docker-compose up --build
```
The API will be available at `http://localhost:8000`. 
*Note: Make sure your trained models are located in the `./models` directory as it is mounted as a volume.*

### 2. Local Installation

```bash
# Clone the repository
git clone https://github.com/AbirSRM/Toxic-VideoGame-chat-predictor-ML-.git
cd Toxic-VideoGame-chat-predictor-ML-

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Usage

### Endpoint: `/api/v1/analyze` (POST)

**Request Body:**
```json
{
  "text": "gg well played!",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "tier_used": "low",
  "label": "Non-Toxic",
  "toxic_prob": 0.0123,
  "predicted_class": 0
}
```

---

## 🧠 Training & Evaluation

The repository includes scripts to train and evaluate the models locally on the Jigsaw Toxicity dataset.

### Training
Train both hardware tiers sequentially:
```bash
python train_both.py
```
Or train a specific tier manually (`high`, `low`, or `distil_high`):
```bash
python train.py --tier low
```

### Evaluation
```bash
python evaluate.py --tier low
```

### Command-line Inference
```bash
python infer.py --tier low --text "Your test chat message here" --threshold 0.5
```

---

## 🛠️ Tech Stack

- **Machine Learning:** PyTorch, HuggingFace `transformers`, `datasets`
- **Models:** `roberta-base` (High Tier), `distilroberta-base` (Low Tier)
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Deployment:** Docker, Docker Compose
