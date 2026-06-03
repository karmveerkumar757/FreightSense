# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import re

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.nlp.intent_classifier import classify_intent_heuristically, re_split_sentences

def main():
    csv_path = os.path.join("data", "Indian_Freight_Delivery_Instructions_Master_400.csv")
    output_path = os.path.join("data", "intent_training_data.csv")

    if not os.path.exists(csv_path):
        print(f"❌ Error: Dataset file not found at: {csv_path}")
        return

    print(f"📊 Loading master dataset: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=3)
    print(f"✅ Loaded {len(df)} raw rows successfully.")

    sentences_list = []
    labels_list = []

    for idx, row in df.iterrows():
        text = str(row['Instruction (Hinglish)']).strip()
        if not text or text == "nan":
            continue

        # Split instruction into individual sentences
        splits = re_split_sentences(text)
        for sent in splits:
            sent_clean = sent.strip()
            # Ignore very short noises
            if len(sent_clean) > 3:
                label = classify_intent_heuristically(sent_clean)
                if label != "UNKNOWN":
                    sentences_list.append(sent_clean)
                    labels_list.append(label)

    # Save to CSV
    out_df = pd.DataFrame({
        "sentence": sentences_list,
        "label": labels_list
    })
    
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"🎉 Generated {len(out_df)} labeled sentences successfully!")
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    main()
