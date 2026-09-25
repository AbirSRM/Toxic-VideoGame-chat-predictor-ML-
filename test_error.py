import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]]), "labels": torch.tensor([1])}
labels = inputs.pop("labels")
        print("KEYS:", inputs.keys())
outputs = model(**inputs)
print("SUCCESS!")
