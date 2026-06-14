# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

from src.nlp.bilstm_crf_ner import BiLSTMCRF, FreightNERDataset, MODEL_NAME, ID2LABEL
import pandas as pd
import spacy

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 20
MAX_LEN = 128
PATIENCE = 5
SAVE_DIR = os.path.join("models", "bilstm_crf_ner")

COLUMN_TO_LABEL = {
    'Cargo Type': 'CARGO_TYPE',
    'Route Restrictions': 'ROUTE_CONSTRAINT',
    'Time Constraints': 'TIME_CONSTRAINT',
    'Special Handling': 'SPECIAL_HANDLING',
    'Compliance / Documentation': 'COMPLIANCE_REQ'
}

def find_entity_offsets(text: str, entity_text: str) -> tuple:
    if not isinstance(entity_text, str):
        return None
    cleaned = entity_text.strip()
    if not cleaned:
        return None
    start = text.lower().find(cleaned.lower())
    if start != -1:
        return start, start + len(cleaned)
    return None

def generate_dataset_from_csv():
    csv_path = os.path.join("data", "Indian_Freight_Delivery_Instructions_Master_400.csv")
    if not os.path.exists(csv_path):
        print("❌ Could not find CSV dataset!")
        return []
        
    df = pd.read_csv(csv_path, skiprows=3)
    nlp = spacy.blank("en")
    
    dataset = []
    
    for idx, row in df.iterrows():
        text = str(row['Instruction (Hinglish)']).strip()
        if not text or text == "nan":
            continue
            
        doc = nlp.make_doc(text)
        ents = []
        for col_name, label in COLUMN_TO_LABEL.items():
            if col_name in row and pd.notna(row[col_name]):
                val = str(row[col_name])
                offsets = find_entity_offsets(text, val)
                if offsets:
                    span = doc.char_span(offsets[0], offsets[1], label=label, alignment_mode="contract")
                    if span:
                        ents.append(span)
                        
        try:
            # We must filter overlapping entities
            filtered_ents = spacy.util.filter_spans(ents)
            doc.ents = filtered_ents
            
            tokens = [t.text for t in doc]
            labels = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ != "O" else "O" for t in doc]
            
            dataset.append({"tokens": tokens, "labels": labels})
        except Exception as e:
            pass
            
    return dataset

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    # Progress bar for epoch
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass calculates NLL loss
        loss = model(input_ids, attention_mask, labels=labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get Viterbi paths
            predictions = model(input_ids, attention_mask)
            
            # Align labels back
            labels = labels.cpu().numpy()
            for i in range(len(labels)):
                valid_mask = labels[i] != -100
                true_labels = [ID2LABEL[l] for l in labels[i][valid_mask]]
                # Viterbi decoding returns variable length lists depending on mask
                # So we just take the predictions for valid tokens
                # Note: The CRF decode method inherently uses the attention mask
                # to return lists of the correct length (ignoring trailing padding).
                # But we also need to ignore subwords (-100).
                
                # predictions[i] has length equal to the unpadded sequence
                # labels[i] has length MAX_LEN
                pred_seq = []
                for p, l in zip(predictions[i], labels[i]):
                    if l != -100:
                        pred_seq.append(ID2LABEL[p])
                        
                all_labels.append(true_labels)
                all_preds.append(pred_seq)
                
    # Calculate strict entity-level metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    return f1, precision, recall, all_labels, all_preds

def main():
    print("="*50)
    print("🧠 Initiating Bi-LSTM-CRF NER Training Pipeline")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Prepare Data
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load data dynamically from CSV
    print("🔄 Building training dataset from CSV...")
    training_data = generate_dataset_from_csv()
    
    if len(training_data) == 0:
        print("❌ Dataset generation failed. Aborting training.")
        return
        
    print(f"Using dataset size: {len(training_data)}")
    
    dataset = FreightNERDataset(training_data, tokenizer, max_len=MAX_LEN)
    
    # 80/10/10 split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Handle extremely small datasets (like the sample)
    if train_size == 0:
        train_size, val_size, test_size = len(dataset), 0, 0
    
    if val_size > 0 and test_size > 0:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    else:
        # Fallback for demo dummy data
        train_dataset = val_dataset = test_dataset = dataset
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 2. Initialize Model
    model = BiLSTMCRF(freeze_bert=False)
    model.to(device)
    
    # 3. Setup Optimizers with differential learning rates
    # BERT layers need a smaller LR, LSTM+CRF need a larger LR
    bert_params = list(model.bert.parameters())
    custom_params = list(model.bilstm.parameters()) + list(model.hidden2tag.parameters()) + list(model.crf.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': 2e-5},
        {'params': custom_params, 'lr': 1e-3}
    ])
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # 4. Training Loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        val_f1, val_prec, val_rec, _, _ = evaluate(model, val_loader, device)
        print(f"Validation F1: {val_f1:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")
        
        # Early Stopping & Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"🌟 New best model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping triggered after {epoch} epochs.")
                break
                
    # 5. Final Evaluation on Test Set
    print("\n" + "="*50)
    print("📊 Final Test Set Evaluation")
    print("="*50)
    
    # Load best model
    best_model_path = os.path.join(SAVE_DIR, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    test_f1, test_prec, test_rec, all_labels, all_preds = evaluate(model, test_loader, device)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print(f"Overall F1-Score:  {test_f1:.4f}")
    print(f"Overall Precision: {test_prec:.4f}")
    print(f"Overall Recall:    {test_rec:.4f}")

if __name__ == "__main__":
    main()
