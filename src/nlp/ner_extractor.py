# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import spacy
from typing import List, Dict, Any

_nlp = None

def get_ner_model():
    """
    Loads and caches the fine-tuned spaCy NER model.
    Falls back to model-last or a blank model if model-best is missing.
    """
    global _nlp
    if _nlp is None:
        model_paths = [
            os.path.join("models", "ner_model", "model-best"),
            os.path.join("models", "ner_model", "model-last")
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"📦 Loading fine-tuned spaCy model from: {path}")
                try:
                    _nlp = spacy.load(path)
                    print("✅ spaCy NER model loaded successfully.")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load model at {path}: {e}")
                    
        if _nlp is None:
            print("⚠️ Fine-tuned NER model not found. Falling back to a blank English pipeline.")
            _nlp = spacy.blank("en")
            
    return _nlp

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Runs the custom spaCy NER model on the input text and returns a list
    of extracted entities with their labels, text, and indices.
    """
    nlp = get_ner_model()
    doc = nlp(text)
    
    extracted = []
    for ent in doc.ents:
        extracted.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
        
    return extracted
