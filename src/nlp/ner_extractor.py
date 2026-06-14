# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
try:
    import spacy
except ImportError:
    spacy = None
import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any
from collections import defaultdict

# Global singletons
_nlp_spacy = None
_bilstm_model = None
_tokenizer = None

def get_bilstm_crf_model():
    """
    Attempts to load the custom PyTorch Bi-LSTM-CRF model.
    Returns (model, tokenizer) if successful, else (None, None).
    """
    global _bilstm_model, _tokenizer
    
    if _bilstm_model is not None and _tokenizer is not None:
        return _bilstm_model, _tokenizer
        
    model_path = os.path.join("models", "bilstm_crf_ner", "best_model.pt")
    if os.path.exists(model_path):
        print(f"📦 Loading fine-tuned Bi-LSTM-CRF model from: {model_path}")
        try:
            from src.nlp.bilstm_crf_ner import BiLSTMCRF, MODEL_NAME
            
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _bilstm_model = BiLSTMCRF(freeze_bert=False)
            
            # Load weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _bilstm_model.load_state_dict(torch.load(model_path, map_location=device))
            _bilstm_model.to(device)
            _bilstm_model.eval()
            
            print("✅ Bi-LSTM-CRF NER model loaded successfully.")
            return _bilstm_model, _tokenizer
        except Exception as e:
            print(f"⚠️ Failed to load Bi-LSTM-CRF model: {e}")
            
    return None, None

def get_spacy_fallback():
    """
    Loads and caches the fine-tuned spaCy NER model as a fallback.
    """
    global _nlp_spacy
    if spacy is None:
        print("⚠️ spaCy is not installed. NER fallback will return empty results.")
        return None
        
    if _nlp_spacy is None:
        model_paths = [
            os.path.join("models", "ner_model", "model-best"),
            os.path.join("models", "ner_model", "model-last")
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"📦 Loading fallback spaCy model from: {path}")
                try:
                    _nlp_spacy = spacy.load(path)
                    return _nlp_spacy
                except Exception as e:
                    pass
                    
        print("⚠️ No custom NER models found. Falling back to base English pipeline.")
        try:
            _nlp_spacy = spacy.load("en_core_web_sm")
        except:
            _nlp_spacy = spacy.blank("en")
            
    return _nlp_spacy

def parse_bilstm_predictions(predictions: List[tuple]) -> Dict[str, Any]:
    """
    Converts BIO tags from the Bi-LSTM-CRF model into the required dictionary format.
    """
    result = {
        "cargo_type": [],
        "time_constraints": [],
        "route_constraints": [],
        "special_handling": [],
        "compliance_reqs": [],
        "locations": [],
        "vehicle_type": [],
        "raw_predictions": predictions
    }
    
    current_entity = []
    current_label = None
    
    def save_entity():
        if not current_entity or not current_label:
            return
            
        entity_text = " ".join(current_entity)
        
        if current_label == "CARGO_TYPE":
            result["cargo_type"].append(entity_text)
        elif current_label == "TIME_CONSTRAINT":
            result["time_constraints"].append(entity_text)
        elif current_label == "ROUTE_CONSTRAINT":
            result["route_constraints"].append(entity_text)
        elif current_label == "SPECIAL_HANDLING":
            result["special_handling"].append(entity_text)
        elif current_label == "COMPLIANCE_REQ":
            result["compliance_reqs"].append(entity_text)
        elif current_label == "LOCATION":
            result["locations"].append(entity_text)
        elif current_label == "VEHICLE_TYPE":
            result["vehicle_type"].append(entity_text)
            
    for word, bio_tag in predictions:
        if bio_tag == "O":
            save_entity()
            current_entity = []
            current_label = None
        elif bio_tag.startswith("B-"):
            save_entity()
            current_entity = [word]
            current_label = bio_tag[2:]
        elif bio_tag.startswith("I-"):
            if current_label == bio_tag[2:]:
                current_entity.append(word)
            else:
                # Invalid transition, treat as B-
                save_entity()
                current_entity = [word]
                current_label = bio_tag[2:]
                
    # Save last entity if exists
    save_entity()
    
    return result

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extracts entities using the Bi-LSTM-CRF model and supplements with spaCy for locations.
    Returns a unified dictionary format.
    """
    result = {
        "cargo_type": [],
        "time_constraints": [],
        "route_constraints": [],
        "special_handling": [],
        "compliance_reqs": [],
        "locations": [],
        "vehicle_type": [],
        "raw_predictions": [] 
    }
    
    # 1. Primary Extraction: PyTorch Bi-LSTM-CRF
    model, tokenizer = get_bilstm_crf_model()
    if model and tokenizer:
        try:
            predictions = model.predict(text, tokenizer)
            bilstm_res = parse_bilstm_predictions(predictions)
            for k in result:
                if k in bilstm_res and bilstm_res[k]:
                    result[k].extend(bilstm_res[k])
        except Exception as e:
            print(f"⚠️ Bi-LSTM prediction error: {e}")
            
    # 2. Supplementary Extraction: spaCy (for Locations & fallbacks)
    nlp = get_spacy_fallback()
    if nlp is not None:
        try:
            doc = nlp(text)
            hinglish_stops = {"hai", "aur", "ke", "liye", "ko", "se", "tak", "mein", "par", "bhi"}
            
            for ent in doc.ents:
                label = ent.label_
                ent_text = ent.text.strip()
                
                # Filter out noisy Hinglish stop words
                if ent_text.lower() in hinglish_stops or len(ent_text) < 2:
                    continue
                    
                if label == "CARGO" and not result["cargo_type"]:
                    result["cargo_type"].append(ent_text)
                elif label in ["TIME", "DATE"] and not result["time_constraints"]:
                    result["time_constraints"].append(ent_text)
                elif label == "ROUTE" and not result["route_constraints"]:
                    result["route_constraints"].append(ent_text)
                elif label == "HANDLING" and not result["special_handling"]:
                    result["special_handling"].append(ent_text)
                elif label == "COMPLIANCE" and not result["compliance_reqs"]:
                    result["compliance_reqs"].append(ent_text)
                elif label in ["GPE", "LOC", "FAC"]:
                    if ent_text not in result["locations"]:
                        result["locations"].append(ent_text)
        except Exception as e:
            print(f"⚠️ spaCy extraction error: {e}")
            
    # 3. Rule-Based Fallback for Known Cities
    # Since spaCy sometimes misclassifies Indian cities in Hinglish (e.g. 'Bangalore jaana hai' as ORG)
    known_cities = ["delhi", "gurugram", "gurgaon", "mumbai", "pune", "bangalore", "bengaluru", 
                   "chennai", "ahmedabad", "jaipur", "kolkata", "hyderabad", "noida", "ghaziabad"]
    
    text_lower = text.lower()
    for city in known_cities:
        # Check if city is in text as a whole word
        import re
        if re.search(r'\b' + city + r'\b', text_lower):
            # Format nicely (capitalize first letter)
            formatted_city = city.capitalize()
            if formatted_city not in result["locations"]:
                result["locations"].append(formatted_city)
                
    # Deduplicate lists
    for k in result:
        if isinstance(result[k], list) and k != "raw_predictions":
            # Preserve order while deduplicating
            seen = set()
            result[k] = [x for x in result[k] if not (x in seen or seen.add(x))]
            
    return result
