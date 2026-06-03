# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define the classes
CLASSES = ["DEADLINE_SET", "ROUTE_RESTRICTION", "HANDLING_INSTRUCTION", "COMPLIANCE_NOTE", "DRIVER_ALERT", "VEHICLE_REQUIREMENT"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

class LogisticsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    csv_path = os.path.join("data", "intent_training_data.csv")
    output_model_dir = os.path.join("models", "bert_intent")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Training dataset not found at {csv_path}.")
        print("Please generate or place intent_training_data.csv first with 'sentence' and 'label' columns.")
        return
        
    print(f"📊 Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter classes
    df = df[df['label'].isin(CLASSES)].reset_index(drop=True)
    df['label_id'] = df['label'].map(CLASS_TO_ID)
    
    sentences = df['sentence'].tolist()
    labels = df['label_id'].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    model_name = "bert-base-multilingual-cased"
    print(f"📥 Loading pre-trained tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("⚡ Tokenizing datasets...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = LogisticsDataset(train_encodings, train_labels)
    val_dataset = LogisticsDataset(val_encodings, val_labels)
    
    print(f"📥 Loading model architectures: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(CLASSES))
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Training device: {device.upper()}")
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results/bert_checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./results/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("🚀 Starting training loop...")
    trainer.train()
    
    print(f"💾 Saving model to {output_model_dir}...")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("🎉 Training complete and model saved successfully!")

if __name__ == "__main__":
    main()
