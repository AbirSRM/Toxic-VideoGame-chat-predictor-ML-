import torch
import os


class TierConfig:
    """Holds all hyperparameters for a specific hardware tier."""
    def __init__(self, model_name, batch_size, max_len, epochs, lr, use_fp16, grad_checkpoint):
        self.model_name      = model_name
        self.batch_size      = batch_size
        self.max_len         = max_len
        self.epochs          = epochs
        self.lr              = lr
        self.use_fp16        = use_fp16
        self.grad_checkpoint = grad_checkpoint


class Config:
    # ── Hardware ──────────────────────────────────────────────────────────────
    # Automatically picks GPU if available — no manual override needed.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _gpu_available = torch.cuda.is_available()

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR         = "./data"
    TRAIN_PATH       = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH        = os.path.join(DATA_DIR, "test.csv")
    TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.csv")

    # ── Columns ───────────────────────────────────────────────────────────────
    TEXT_COLUMN       = "comment_text"
    LABEL_COLUMN      = "target"       # float [0,1] in train.csv (Jigsaw 2019)
    TEST_LABEL_COLUMN = "toxic"        # binary {0,1,-1} in test_labels.csv (2018)
    NUM_LABELS        = 2

    # ── Tier Definitions ──────────────────────────────────────────────────────
    # Low       -> distilroberta-base  (~82M params) — optimised for CPUs / <=8GB RAM
    # Distil Hi -> distilroberta-base  (~82M params) — high-tier hyperparams on GPU
    # High      -> roberta-base        (~125M params) — uses GPU + FP16 if available
    TIERS = {
        "low": TierConfig(
            model_name      = "distilroberta-base",
            batch_size      = 16,
            max_len         = 128,
            epochs          = 2,
            lr              = 3e-5,
            use_fp16        = _gpu_available,
            grad_checkpoint = False,
        ),
        "distil_high": TierConfig(
            model_name      = "distilroberta-base",
            batch_size      = 8,
            max_len         = 256,
            epochs          = 3,
            lr              = 2e-5,
            use_fp16        = _gpu_available,
            grad_checkpoint = True,
        ),
        "high": TierConfig(
            model_name      = "roberta-base",
            batch_size      = 8,
            max_len         = 256,
            epochs          = 3,
            lr              = 2e-5,
            use_fp16        = _gpu_available,
            grad_checkpoint = True,   # Saves memory during high-tier training
        ),
    }

    @classmethod
    def get_tier(cls, tier: str) -> TierConfig:
        return cls.TIERS.get(tier, cls.TIERS["low"])

    @classmethod
    def output_dir(cls, tier: str) -> str:
        return f"./models/{tier}_tier"