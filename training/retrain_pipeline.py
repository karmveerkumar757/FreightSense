# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feedback.feedback_db import SessionLocal, ShipmentRecord
from src.genai.llm_caller import query_llm_advisory
from src.output.json_validator import clean_llm_json_string

def parse_feedback_with_llm(raw_text: str, feedback_notes: str, original_entities: list, original_intents: list) -> dict:
    """
    Uses the LLM to semantically parse the dispatcher's free-form feedback notes
    and output the corrected named entities and sentence intents in a strict JSON schema.
    """
    prompt = f"""
    You are an MLOps data labeling assistant. A logistics dispatcher has flagged a shipment instruction and provided correction notes.
    Your task is to output the CORRECTED named entities and sentence intents based on the dispatcher's feedback notes.
    
    ORIGINAL TEXT:
    "{raw_text}"
    
    DISPATCHER FEEDBACK NOTES:
    "{feedback_notes}"
    
    ORIGINAL ENTITIES:
    {json.dumps(original_entities, ensure_ascii=False)}
    
    ORIGINAL INTENTS:
    {json.dumps(original_intents, ensure_ascii=False)}
    
    INSTRUCTIONS:
    1. Update the named entities. For entities, identify the exact substring inside the ORIGINAL TEXT, calculate its start and end char indices, and map it to one of: CARGO_TYPE, TIME_CONSTRAINT, ROUTE_CONSTRAINT, SPECIAL_HANDLING, COMPLIANCE_REQ.
    2. Update the sentence intents. Map each sentence to one of: DEADLINE_SET, ROUTE_RESTRICTION, HANDLING_INSTRUCTION, COMPLIANCE_NOTE, DRIVER_ALERT, VEHICLE_REQUIREMENT.
    
    OUTPUT FORMAT:
    You MUST output your response as a valid JSON object ONLY. Do not include markdown code fence wrappers or trailing text.
    
    JSON SCHEMA REQUIRED:
    {{
      "entities": [
        {{
          "text": "exact substring",
          "label": "LABEL",
          "start": 0,
          "end": 10
        }}
      ],
      "intents": [
        {{
          "sentence": "sentence text",
          "intent": "INTENT_LABEL"
        }}
      ]
    }}
    """
    print(f"☁️ Querying LLM to parse dispatcher feedback semantic corrections...")
    try:
        response = query_llm_advisory(prompt)
        cleaned = clean_llm_json_string(response)
        return json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ Failed to parse feedback with LLM: {e}")
        return None

def run_retraining_pipeline():
    print("🔄 Starting MLOps Active Retraining Pipeline...")
    
    # 1. Fetch feedback records from DB
    db = SessionLocal()
    try:
        feedback_records = db.query(ShipmentRecord).filter(ShipmentRecord.has_feedback == True).all()
        if not feedback_records:
            print("ℹ️ No shipment feedback logs found in SQLite database. Nothing to retrain.")
            return
            
        print(f"📈 Found {len(feedback_records)} shipments with dispatcher overrides.")
        
        # 2. Parse overrides and build corrections maps
        ner_corrections = {}
        intent_corrections = {}
        
        for record in feedback_records:
            print(f"\n📦 Processing feedback for Shipment ID: {record.id}")
            orig_ents = json.loads(record.extracted_entities) if record.extracted_entities else []
            orig_intents = json.loads(record.sentence_intents) if record.sentence_intents else []
            
            parsed = parse_feedback_with_llm(
                raw_text=record.raw_input_text,
                feedback_notes=record.feedback_notes,
                original_entities=orig_ents,
                original_intents=orig_intents
            )
            
            if parsed:
                # Add NER correction
                # Map standard spaCy NER format: [start, end, label]
                spacy_ents = []
                for ent in parsed.get("entities", []):
                    spacy_ents.append([ent["start"], ent["end"], ent["label"]])
                ner_corrections[record.raw_input_text] = spacy_ents
                
                # Add intent corrections
                for it in parsed.get("intents", []):
                    intent_corrections[it["sentence"]] = it["intent"]
                    
        # 3. Apply corrections to spaCy NER training data
        ner_data_path = os.path.join("data", "ner_training_data.py")
        if os.path.exists(ner_data_path):
            print(f"\n🏷️ Updating spaCy NER dataset: {ner_data_path}")
            # Import current training data
            try:
                from data.ner_training_data import TRAINING_DATA
            except ImportError:
                TRAINING_DATA = []
                
            # Update matching text values
            updated_count = 0
            for item in TRAINING_DATA:
                text = item["text"]
                if text in ner_corrections:
                    item["entities"] = ner_corrections[text]
                    updated_count += 1
                    
            print(f"✅ Updated {updated_count} existing entries in training data with dispatcher feedback.")
            
            # Save updated ner_training_data.py
            with open(ner_data_path, "w", encoding="utf-8") as f:
                f.write("# -*- coding: utf-8 -*-\n")
                f.write(f"# Auto-updated dataset via MLOps Feedback Loop\n")
                f.write("TRAINING_DATA = ")
                json.dump(TRAINING_DATA, f, indent=4, ensure_ascii=False)
                
            # Re-compile spaCy dataset binaries
            print("⚡ Re-compiling train.spacy and valid.spacy...")
            try:
                # Run train_ner.py script (loads TRAINING_DATA and outputs train.spacy / valid.spacy)
                subprocess.run([sys.executable, "train_ner.py"], check=True)
            except Exception as e:
                print(f"❌ Failed to compile spaCy dataset: {e}")
                return
        else:
            print("⚠️ ner_training_data.py not found. Skipping NER dataset update.")

        # 4. Apply corrections to BERT Intent dataset
        intent_csv_path = os.path.join("data", "intent_training_data.csv")
        if os.path.exists(intent_csv_path):
            print(f"\n🎯 Updating BERT Intent dataset: {intent_csv_path}")
            intent_df = pd.read_csv(intent_csv_path)
            
            updated_intents = 0
            for sentence, label in intent_corrections.items():
                # Find matching sentence
                matches = intent_df[intent_df["sentence"] == sentence]
                if not matches.empty:
                    # Update label
                    intent_df.loc[intent_df["sentence"] == sentence, "label"] = label
                    updated_intents += 1
                else:
                    # Append new label
                    new_row = pd.DataFrame([{"sentence": sentence, "label": label}])
                    intent_df = pd.concat([intent_df, new_row], ignore_index=True)
                    updated_intents += 1
                    
            intent_df.to_csv(intent_csv_path, index=False, encoding='utf-8')
            print(f"✅ Updated/added {updated_intents} sentences in intent dataset.")
        else:
            print("⚠️ intent_training_data.csv not found. Skipping Intent dataset update.")

        # 5. Run spaCy Model Retraining Loop
        print("\n🚀 Retraining custom spaCy NER model...")
        try:
            # We train on CPU for reliability
            subprocess.run([
                sys.executable, "-m", "spacy", "train", "config.cfg", 
                "--output", "models/ner_model", "--gpu-id", "-1"
            ], check=True)
            print("✅ spaCy NER model retraining finished successfully!")
        except Exception as e:
            print(f"❌ spaCy NER retraining failed: {e}")

        # 6. Run BERT Model Retraining Loop
        print("\n🚀 Retraining BERT Intent Classification model...")
        try:
            subprocess.run([sys.executable, "training/finetune_bert.py"], check=True)
            print("✅ BERT Intent model retraining finished successfully!")
        except Exception as e:
            print(f"❌ BERT retraining failed: {e}")
            
        print("\n🎉 FreightSense Active Retraining Loop Complete!")
        
    finally:
        db.close()

if __name__ == "__main__":
    run_retraining_pipeline()
