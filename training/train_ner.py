# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from sklearn.model_selection import train_test_split

COLUMN_TO_LABEL = {
    'Cargo Type': 'CARGO_TYPE',
    'Route Restrictions': 'ROUTE_CONSTRAINT',
    'Time Constraints': 'TIME_CONSTRAINT',
    'Special Handling': 'SPECIAL_HANDLING',
    'Compliance / Documentation': 'COMPLIANCE_REQ'
}

def find_entity_offsets(text: str, entity_text: str) -> tuple:
    """
    Finds the exact start and end char indices of an entity substring inside the instruction.
    Uses case-insensitive matching and handles surrounding whitespace cleaning.
    """
    if not isinstance(entity_text, str):
        return None
        
    cleaned_entity = entity_text.strip()
    if not cleaned_entity:
        return None
        
    start_idx = text.lower().find(cleaned_entity.lower())
    if start_idx != -1:
        return start_idx, start_idx + len(cleaned_entity)
        
    return None

def process_csv_dataset():
    csv_path = os.path.join("data", "Indian_Freight_Delivery_Instructions_Master_400.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Dataset file not found at: {csv_path}")
        print("Please place the master CSV file inside the 'data/' directory first.")
        return
        
    print(f"📊 Loading master dataset: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=3)
    print(f"✅ Loaded {len(df)} raw rows successfully.")
    
    nlp = spacy.blank("en")
    docs = []
    skipped_entities = 0
    total_entities = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📦 Processing sentences"):
        text = str(row['Instruction (Hinglish)']).strip()
        if not text:
            continue
            
        doc = nlp.make_doc(text)
        ents = []
        
        # Extract entities for each column mapping
        for col_name, label in COLUMN_TO_LABEL.items():
            if col_name in row and pd.notna(row[col_name]):
                val = str(row[col_name])
                offsets = find_entity_offsets(text, val)
                if offsets:
                    total_entities += 1
                    start, end = offsets
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is not None:
                        ents.append(span)
                    else:
                        skipped_entities += 1
                        
        try:
            doc.ents = ents
            docs.append(doc)
        except Exception as e:
            print(f"⚠️ Error setting entities on row {idx}: {e}")
            
    print(f"\n📊 Extracted {total_entities} entities. Skipped {skipped_entities} due to alignment boundary issues.")
    
    # 80/20 Train/Validation Split
    train_docs, valid_docs = train_test_split(docs, test_size=0.2, random_state=42)
    
    train_db = DocBin(docs=train_docs)
    valid_db = DocBin(docs=valid_docs)
    
    train_db.to_disk("train.spacy")
    valid_db.to_disk("valid.spacy")
    print(f"💾 Saved {len(train_docs)} training records to 'train.spacy'")
    print(f"💾 Saved {len(valid_docs)} validation records to 'valid.spacy'")
    print("🎉 Dataset conversion and token-alignment complete!")

if __name__ == "__main__":
    process_csv_dataset()
