import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
train_dataset = Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5, 6]], "attention_mask": [[1, 1, 1], [1, 1, 1]], "labels": [0, 1]})

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        print("KEYS IN INPUTS:", inputs.keys())
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        return torch.tensor(0.0, requires_grad=True)

trainer = MyTrainer(
    model=model,
    args=TrainingArguments(output_dir="./tmp", max_steps=1, per_device_train_batch_size=2),
    train_dataset=train_dataset
)
trainer.train()
