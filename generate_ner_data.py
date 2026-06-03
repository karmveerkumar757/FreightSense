import os
import json
import time
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# 1. Initialize the correct modern SDK client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY missing from .env file.")
client = genai.Client(api_key=api_key)

# 2. Use Pydantic to strictly define the structured data format.
# This forces the API to validate the output structure natively before returning it!
class EntityAnnotation(BaseModel):
    text: str = Field(description="The exact text snippet corresponding to the entity.")
    label: str = Field(description="Must be exactly one of: CARGO_TYPE, TIME_CONSTRAINT, ROUTE_CONSTRAINT, SPECIAL_HANDLING, COMPLIANCE_REQ")
    start_char_idx: int = Field(description="The starting character index of the entity text snippet within the full instruction string.")
    end_char_idx: int = Field(description="The ending character index of the entity text snippet within the full instruction string.")

class ShipmentInstruction(BaseModel):
    instruction_text: str = Field(description="The full Hinglish logistics text generated.")
    entities: List[EntityAnnotation] = Field(description="List of all extracted logistics entities found in the text.")

class BatchLogisticsDataset(BaseModel):
    dataset: List[ShipmentInstruction]

def find_closest_occurrence(text, ent_text, suggested_start):
    cleaned = ent_text.strip()
    if not cleaned:
        return None
    
    best_start = -1
    min_dist = float('inf')
    
    text_lower = text.lower()
    cleaned_lower = cleaned.lower()
    
    start_pos = 0
    while True:
        pos = text_lower.find(cleaned_lower, start_pos)
        if pos == -1:
            break
        dist = abs(pos - suggested_start)
        if dist < min_dist:
            min_dist = dist
            best_start = pos
        start_pos = pos + 1
        
    if best_start != -1:
        return best_start, best_start + len(cleaned)
    return None

def generate_batch(batch_num):
    print(f"🔄 Processing Batch {batch_num}/20 via Gemini Native Structured Output...")
    
    prompt = """
    Generate 20 completely unique, highly realistic Indian freight delivery instructions.
    
    CRITICAL INPUT VARIABLES:
    1. Mix Hindi and English words naturally (Hinglish) like actual Indian truck drivers and dispatchers speak (e.g., 'kal subah tak deliver karna hai', 'NH-48 avoid karo').
    2. Vary sentence structures, urgency levels, target cities, and vehicle parameters.
    3. Include entities from these 5 labels:
       - CARGO_TYPE: items being shipped (e.g., electronics, medicine, frozen food)
       - TIME_CONSTRAINT: deadlines, windows (e.g., before 8 AM, 2 din me, urgent tonight)
       - ROUTE_CONSTRAINT: roads to take or avoid (e.g., avoid Ring Road, NH-44 se jao)
       - SPECIAL_HANDLING: storage rules (e.g., refrigerated, keep dry, fragile)
       - COMPLIANCE_REQ: documentation, permits (e.g., e-Way Bill check, Form 9 copy)
    
    For every generated instruction sentence, carefully calculate the exact start and end character positions for each entity found in your text.
    """

    retries = 5
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            # Use the explicit .models.generate_content route with a schema configuration
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=dict(
                    response_mime_type="application/json",
                    response_schema=BatchLogisticsDataset,
                    temperature=0.7
                )
            )
            
            # Parse output safely via JSON 
            raw_json = json.loads(response.text)
            
            # Remap to match standard spaCy NER format: [start, end, label]
            formatted_batch = []
            for item in raw_json.get("dataset", []):
                spacy_entities = []
                text = item["instruction_text"]
                for ent in item.get("entities", []):
                    aligned = find_closest_occurrence(text, ent["text"], ent["start_char_idx"])
                    if aligned is not None:
                        spacy_entities.append([aligned[0], aligned[1], ent["label"]])
                    else:
                        spacy_entities.append([ent["start_char_idx"], ent["end_char_idx"], ent["label"]])
                
                formatted_batch.append({
                    "text": text,
                    "entities": spacy_entities
                })
                
            print(f"✅ Batch {batch_num}/20 successfully generated ({len(formatted_batch)} examples compiled).")
            return formatted_batch

        except Exception as e:
            if attempt == retries:
                print(f"⚠️ Error processing batch {batch_num} after {retries} attempts: {e}")
                return []
            
            # Check for rate limit or temp unavailable and back off
            err_str = str(e).lower()
            backoff = delay * (2 ** (attempt - 1))
            if "429" in err_str or "resource_exhausted" in err_str:
                backoff = max(backoff, 20.0) # sleep longer for rate limits
                print(f"⏳ Rate limit hit on Batch {batch_num} (attempt {attempt}/{retries}). Sleeping {backoff:.1f}s before retry...")
            else:
                print(f"⏳ Temp error ({e}) on Batch {batch_num} (attempt {attempt}/{retries}). Sleeping {backoff:.1f}s before retry...")
            time.sleep(backoff)

def main():
    total_dataset = []
    
    # Run 20 quick loop iterations to get our 400 sample targets
    for i in range(1, 21):
        batch = generate_batch(i)
        total_dataset.extend(batch)
        time.sleep(12.0) # Respect free tier rate limits (5 requests per minute = 1 request every 12 seconds)
        
    print(f"\n📊 Master Extraction Complete. Dataset contains {len(total_dataset)} items.")
    
    output_path = "data/ner_training_data.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write(f"# Auto-generated dataset via Gemini Schema Supervision\n")
        f.write("TRAINING_DATA = ")
        json.dump(total_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"🎉 Master dataset compiled successfully and written to: {output_path}")

if __name__ == "__main__":
    main()