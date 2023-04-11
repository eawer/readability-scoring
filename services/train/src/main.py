import torch
import pandas as pd
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, TrainingArguments, Trainer, BertForSequenceClassification

from sklearn.model_selection import train_test_split


SEED = 0
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(tokenizer, examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=300, return_tensors="pt", return_attention_mask=True)

def create_dataset(train, evaluation):
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict(train.to_dict(orient='list'), split="train")
    dataset['eval'] = Dataset.from_dict(evaluation.to_dict(orient='list'), split="eval")

    return dataset

class BertRegressorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = torch.sqrt(nn.functional.mse_loss(outputs["logits"].squeeze(), labels))

        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions).squeeze()
    mse = torch.mean((preds - labels) ** 2)
    rmse = torch.sqrt(mse)

    return {
        'rmse': rmse,
    }


if __name__ == "__main__":
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True)
    bert_model.to(DEVICE)

    train_df = pd.read_csv("/input/train.csv.zip", usecols=[3, 4])
    train_df = train_df.rename(columns={'excerpt': 'text', 'target': 'labels'})
    train_df, eval_df = train_test_split(train_df, test_size=0.1, stratify=pd.cut(train_df["labels"], 5), random_state=SEED)

    dataset = create_dataset(train_df, eval_df)
    tokenized_dataset = dataset.map(lambda examples: tokenize(tokenizer, examples), batched=True, batch_size=64, remove_columns="text")
    tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask'], output_all_columns=True)

    training_args = TrainingArguments(
        output_dir="trainer", 
        evaluation_strategy="epoch", 
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3.0,
        learning_rate=1e-5,
        optim="adamw_torch",
    )
    trainer = BertRegressorTrainer(
        model=bert_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'].shuffle(seed=SEED),
        eval_dataset=tokenized_dataset['eval'].shuffle(seed=SEED),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model("/model")
    tokenizer.save_pretrained("/tokenizer")
