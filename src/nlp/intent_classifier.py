# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
from typing import List, Dict, Any

# Heuristic intent keywords
INTENT_KEYWORDS = {
    "DEADLINE_SET": ["before", "tak", "deliver", "baje", "pm", "am", "urgent", "shaam", "subah", "noon", "hours", "hrs", "time", "clock"],
    "ROUTE_RESTRICTION": ["avoid", "highway", "nh-", "route", "bypass", "road", "expressway", "flyover", "tunnel", "ban", "mat jaana", "se hi aana"],
    "HANDLING_INSTRUCTION": ["refrigerated", "keep", "frozen", "temperature", "cold", "fragile", "handle", "dry", "ice", "cool", "careful", "dhyan se", "damage"],
    "COMPLIANCE_NOTE": ["e-way", "bill", "permit", "invoice", "paper", "document", "gst", "challan", "msds", "compliance", "sign", "copy", "papers"],
    "DRIVER_ALERT": ["customer", "call", "phone", "number", "gate", "guard", "alert", "driver", "careful", "warning", "ruke", "stay"],
    "VEHICLE_REQUIREMENT": ["truck", "flatbed", "container", "open", "capacity", "vehicle", "axle", "load", "ft", "feet", "tempo", "trolley", "dumper"]
}

def classify_intent_heuristically(sentence: str) -> str:
    """
    Categorizes the intent of a logistics sentence based on keyword matching.
    """
    sentence_lower = sentence.lower()
    best_intent = "UNKNOWN"
    max_matches = 0
    
    for intent, keywords in INTENT_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in sentence_lower)
        if matches > max_matches:
            max_matches = matches
            best_intent = intent
            
    # Fallback default heuristics based on common words if still unknown
    if best_intent == "UNKNOWN":
        if any(w in sentence_lower for w in ["kal", "aaj", "parson", "baje"]):
            return "DEADLINE_SET"
        if any(w in sentence_lower for w in ["avoid", "nh", "road"]):
            return "ROUTE_RESTRICTION"
            
    return best_intent

# Transformers model caching
_tokenizer = None
_model = None

def get_bert_classifier():
    """
    Attempts to load a fine-tuned BERT model if it has been compiled.
    """
    global _tokenizer, _model
    model_dir = os.path.join("models", "bert_intent")
    if os.path.exists(model_dir) and any(os.listdir(model_dir)):
        if _model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                print(f"📦 Loading fine-tuned BERT intent classifier from: {model_dir}")
                _tokenizer = AutoTokenizer.from_pretrained(model_dir)
                _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            except Exception as e:
                print(f"⚠️ Failed to load BERT model: {e}")
        return _tokenizer, _model
    return None, None

def classify_sentence_intent(sentence: str) -> str:
    """
    Classifies the intent of a single sentence. Uses the BERT classifier if available,
    otherwise falls back to the robust keyword-based heuristic classifier.
    """
    tokenizer, model = get_bert_classifier()
    if model is not None and tokenizer is not None:
        try:
            import torch
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=1).item()
            
            # Map predictions to labels
            classes = ["DEADLINE_SET", "ROUTE_RESTRICTION", "HANDLING_INSTRUCTION", "COMPLIANCE_NOTE", "DRIVER_ALERT", "VEHICLE_REQUIREMENT"]
            if pred_id < len(classes):
                return classes[pred_id]
        except Exception as e:
            print(f"⚠️ BERT inference failed: {e}. Falling back to heuristic classifier.")
            
    return classify_intent_heuristically(sentence)

def get_sentence_intents(text: str) -> List[Dict[str, str]]:
    """
    Splits the full text into individual sentences and classifies the intent of each.
    """
    # Simple sentence splitter on common delimiters
    sentences = re_split_sentences(text)
    
    results = []
    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) > 3:
            results.append({
                "sentence": sent_clean,
                "intent": classify_sentence_intent(sent_clean)
            })
    return results

def re_split_sentences(text: str) -> List[str]:
    """
    Splits text by full stops, exclamation marks, or question marks.
    """
    import re
    # Split by '.', '!', '?' but avoid splitting abbreviations like NH-48 or e.g.
    # A simple regex split is sufficient for this domain
    splits = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in splits if s]
