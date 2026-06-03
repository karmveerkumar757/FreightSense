# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import os

# Import the dataset we generated
# If running on Colab, ensure ner_training_data.py is in the same directory or python path
try:
    from data.ner_training_data import TRAINING_DATA
except ImportError:
    from ner_training_data import TRAINING_DATA

def convert_to_spacy_format():
    print(f"📦 Converting {len(TRAINING_DATA)} examples into spaCy DocBin format...")
    nlp = spacy.blank("en") # Create a blank English/multilingual base pipeline
    db = DocBin()
    
    skipped_count = 0
    for item in tqdm(TRAINING_DATA):
        text = item["text"]
        entities = item["entities"]
        
        doc = nlp.make_doc(text)
        ents = []
        
        for start, end, label in entities:
            # Create a character span and validate alignment
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped_count += 1
                continue
            ents.append(span)
            
        try:
            doc.ents = ents
            db.add(doc)
        except Exception:
            skipped_count += 1
            
    print(f"⚠️ Skipped {skipped_count} misaligned entity boundaries during compilation.")
    
    # Split into 80% Training and 20% Evaluation sets
    docs = list(db.get_docs(nlp.vocab))
    split_idx = int(len(docs) * 0.8)
    
    train_db = DocBin(docs=docs[:split_idx])
    valid_db = DocBin(docs=docs[split_idx:])
    
    # Save binaries out for the spaCy compiler
    train_db.to_disk("train.spacy")
    valid_db.to_disk("valid.spacy")
    print("💾 Created train.spacy and valid.spacy successfully!")

if __name__ == "__main__":
    convert_to_spacy_format()