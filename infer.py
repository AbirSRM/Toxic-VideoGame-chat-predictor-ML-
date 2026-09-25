import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config

# ── Module-level cache: avoids reloading the model on every call ──────────────
_cache: dict = {}


def load_model(tier: str):
    if tier not in _cache:
        model_path = f"./models/{tier}_tier/final_model"
        cfg        = Config.get_tier(tier)
        device     = Config.DEVICE

        print(f"[+] Loading '{tier.upper()}' model from {model_path} → {device.upper()}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        _cache[tier] = (model, tokenizer, device, cfg)

    return _cache[tier]


def predict(text: str, tier: str, threshold: float = 0.5) -> dict:
    """
    Classify a single text string.

    Args:
        text      : Input text to classify.
        tier      : 'low' (DistilRoBERTa) or 'high' (RoBERTa).
        threshold : Probability cutoff for 'Toxic'. Default 0.5.
                    Lower → more sensitive (catches more toxicity, more false positives).
                    Higher → more conservative (fewer false positives, may miss some).

    Returns:
        dict with keys: label, toxic_prob, predicted_class
    """
    model, tokenizer, device, cfg = load_model(tier)

    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation     = True,
        max_length     = cfg.max_len,
    ).to(device)

    with torch.no_grad():
        outputs    = model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=-1)[0]
        toxic_prob = probs[1].item()

    predicted_class = 1 if toxic_prob >= threshold else 0
    label           = "Toxic" if predicted_class == 1 else "Non-Toxic"
    confidence      = toxic_prob if predicted_class == 1 else probs[0].item()

    print(
        f"Tier: {tier.upper()} | "
        f"Prediction: {label} | "
        f"Toxic Prob: {toxic_prob:.4f} | "
        f"Confidence: {confidence:.4f}"
    )
    return {"label": label, "toxic_prob": toxic_prob, "predicted_class": predicted_class}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single text input")
    parser.add_argument("--tier",      type=str,   choices=["high", "distil_high", "low"], default="low")
    parser.add_argument("--text",      type=str,   required=True, help="Text to classify")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Toxic probability threshold (default: 0.5)")
    args = parser.parse_args()
    predict(args.text, args.tier, args.threshold)