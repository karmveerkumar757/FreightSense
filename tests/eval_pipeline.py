# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
from src.ingestion.text_cleaner import clean_text
from src.nlp.ner_extractor import extract_entities
from src.nlp.intent_classifier import get_sentence_intents
from src.genai.rag_retriever import retrieve_context
from src.genai.prompt_builder import build_compliance_prompt
from src.genai.llm_caller import query_llm_advisory
from src.output.json_validator import parse_and_validate_advisory
from src.output.alert_generator import generate_driver_whatsapp_alert

def run_test_pipeline(input_text: str, shipment_id: str = "SHP-TEST-001"):
    print("\n" + "="*50)
    print(f"🚦 Running End-to-End Pipeline for {shipment_id}")
    print("="*50)
    
    # 1. Normalization
    print("\n🧹 Step 1: Cleaning input text...")
    cleaned = clean_text(input_text)
    print(f"  Cleaned text: {cleaned}")
    
    # 2. NLP Named Entity Recognition
    print("\n🏷️ Step 2: Running spaCy Named Entity Recognition...")
    entities = extract_entities(cleaned)
    print(f"  Extracted Entities: {json.dumps(entities, indent=2, ensure_ascii=False)}")
    
    # 3. NLP Intent Classification
    print("\n🎯 Step 3: Running Intent Classifier...")
    intents = get_sentence_intents(cleaned)
    print(f"  Sentence Intents: {json.dumps(intents, indent=2, ensure_ascii=False)}")
    
    # 4. RAG Retrieval
    print("\n🔍 Step 4: Querying ChromaDB for Regulations...")
    # Query ChromaDB using extracted entities or raw text
    retrieved_chunks = retrieve_context(cleaned, n_results=3)
    print(f"  Retrieved {len(retrieved_chunks)} relevant regulation chunks.")
    for idx, chunk in enumerate(retrieved_chunks):
        print(f"    Chunk #{idx+1} (first 100 chars): {chunk[:100]}...")
        
    # 5. Build LLM Prompt
    print("\n📝 Step 5: Formatting compliance advisory prompt...")
    nlp_results = {
        "text": cleaned,
        "entities": entities,
        "intents": intents
    }
    prompt = build_compliance_prompt(nlp_results, retrieved_chunks)
    
    # 6. Call LLM
    print("\n🤖 Step 6: Querying LLM (Gemini/Ollama)...")
    try:
        raw_advisory = query_llm_advisory(prompt)
        print("  LLM responded successfully.")
    except Exception as e:
        print(f"❌ LLM query failed: {e}")
        return
        
    # 7. Validate Output JSON
    print("\n🛡️ Step 7: Validating LLM output JSON...")
    try:
        validated_advisory = parse_and_validate_advisory(raw_advisory)
        print("  Advisory JSON validated successfully.")
        print(f"  Overall Risk Rating: {validated_advisory.get('overall_risk', 'UNKNOWN').upper()}")
        print(f"  Advisory: {validated_advisory.get('plain_english_advisory')}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return
        
    # 8. Generate Driver Alert
    print("\n📢 Step 8: Generating Driver WhatsApp Alert...")
    driver_alert = generate_driver_whatsapp_alert(shipment_id, validated_advisory)
    print("-"*40)
    print(driver_alert)
    print("-"*40)
    print("\n✅ End-to-End test pipeline ran successfully!")

if __name__ == "__main__":
    sample_text = (
        "Urgent electronics shipment Bangalore se Gurgaon bhejna hai kal subah 10 baje se pehle. "
        "NH-8 bypass use karna, avoid Delhi inner ring road due to truck ban. "
        "Make sure fragile items have bubble wrap. e-Way Bill check slip is attached, valid until 8 PM."
    )
    run_test_pipeline(sample_text)
