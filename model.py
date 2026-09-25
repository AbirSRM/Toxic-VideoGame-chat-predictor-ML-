from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config


def get_model_and_tokenizer(tier: str):
    cfg = Config.get_tier(tier)
    print(f"[+] Loading '{tier.upper()}' tier model -> {cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=Config.NUM_LABELS,
    )
    return model, tokenizer