"""
Sequential two-stage training pipeline for ONLYPosiChat.
  Stage 1: DistilRoBERTa-base  with high-tier hyperparameters
  Stage 2: RoBERTa-base        with high-tier hyperparameters
"""
import time
import traceback
from train import run_training

STAGES = [
    ("distil_high", "DistilRoBERTa (high-tier params)"),
    ("high",        "RoBERTa-base  (high-tier params)"),
]

if __name__ == "__main__":
    for tier, desc in STAGES:
        print("\n" + "=" * 70)
        print(f"  STAGE: {desc}")
        print(f"  Tier : --tier {tier}")
        print("=" * 70 + "\n")

        t0 = time.time()
        try:
            run_training(tier, sample_size=None)   # full dataset
        except Exception:
            traceback.print_exc()
            print(f"\n[!!] Stage '{tier}' FAILED. Continuing to next stage...\n")
            continue

        elapsed = time.time() - t0
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f"\n[OK] Stage '{tier}' completed in {int(hrs)}h {int(mins)}m {int(secs)}s\n")

    print("\n" + "=" * 70)
    print("  ALL STAGES COMPLETE")
    print("=" * 70)
