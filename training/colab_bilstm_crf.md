# FreightSense: Bi-LSTM-CRF NER Training on Google Colab

This document contains the exact code to copy-paste into a Google Colab notebook (using a free T4 GPU) to train your new Bi-LSTM-CRF NER model.

## 1. Setup & Installation
Create a new cell and run:
```python
!pip install transformers torch pytorch-crf seqeval google-generativeai
```

## 2. Clone the Repository
Create a new cell to mount your drive or clone the repository (adjust if you pushed it to GitHub):
```python
# If your project is on GitHub:
# !git clone https://github.com/yourusername/FreightSense_Project.git
# %cd FreightSense_Project

# Alternatively, just ensure you upload your src/ and training/ folders to Colab.
```

## 3. Synthetic Data Generation (Gemini API)
Create a new cell. This script uses the free Gemini 1.5 Flash API to generate 400 training sentences in Hinglish with precise BIO tags.

```python
import google.generativeai as genai
import json
import os

# Configure your FREE Gemini API Key (Get from aistudio.google.com)
genai.configure(api_key="YOUR_FREE_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

prompt = """
Generate 400 highly diverse, synthetic logistics and freight dispatch instructions in a mix of Hindi and English (Hinglish). 
Format the output as a strict JSON list of dictionaries. Each dictionary must have two keys: "tokens" (a list of words) and "labels" (a list of corresponding BIO tags).

The allowed BIO tags are:
O, B-CARGO_TYPE, I-CARGO_TYPE, B-TIME_CONSTRAINT, I-TIME_CONSTRAINT, B-ROUTE_CONSTRAINT, I-ROUTE_CONSTRAINT, B-SPECIAL_HANDLING, I-SPECIAL_HANDLING, B-COMPLIANCE_REQ, I-COMPLIANCE_REQ, B-LOCATION, I-LOCATION, B-VEHICLE_TYPE, I-VEHICLE_TYPE

Example Output format:
[
  {
    "tokens": ["Bhai", "medicine", "ka", "load", "Pune", "before", "5", "PM", "bhejna", "hai"],
    "labels": ["O", "B-CARGO_TYPE", "O", "O", "B-LOCATION", "B-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "O", "O"]
  }
]
Output ONLY valid JSON, nothing else. No markdown formatting.
"""

print("Calling Gemini API to generate 400 synthetic samples...")
response = model.generate_content(prompt)
raw_text = response.text.replace("```json", "").replace("```", "").strip()

try:
    dataset = json.loads(raw_text)
    print(f"Successfully generated {len(dataset)} training samples.")
    
    # Save the generated dataset so the training script can load it
    os.makedirs("data", exist_ok=True)
    with open("data/ner_training_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
except Exception as e:
    print("Failed to parse JSON from Gemini output. Try running again.", e)
```

## 4. Train the Model
*Note: Before running this cell, make sure you update `training/train_bilstm_crf.py` to load `data/ner_training_data.json` instead of the `SAMPLE_TRAINING_DATA` array.*

Create a new cell and run the training loop:
```python
!python training/train_bilstm_crf.py
```

## 5. Download the Trained Weights
After training completes (should take about 10-15 minutes on a T4 GPU), download your best model.
```python
from google.colab import files
files.download("models/bilstm_crf_ner/best_model.pt")
```
After downloading, place `best_model.pt` into your local project directory at `models/bilstm_crf_ner/best_model.pt` so `src/nlp/ner_extractor.py` can load it!
