# -*- coding: utf-8 -*-
import os
import sys

# Force Hugging Face offline mode to avoid slow/hanging online checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import uuid
import shutil
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Pipeline Imports
from src.ingestion.text_cleaner import clean_text
from src.ingestion.ocr import extract_text_from_pdf, extract_text_from_image
from src.ingestion.asr import transcribe_audio
from src.nlp.ner_extractor import extract_entities
from src.nlp.intent_classifier import get_sentence_intents
from src.genai.rag_retriever import retrieve_context
from src.genai.prompt_builder import build_compliance_prompt
from src.genai.llm_caller import query_llm_advisory
from src.output.json_validator import parse_and_validate_advisory
from src.output.alert_generator import generate_driver_whatsapp_alert
from src.feedback.feedback_db import save_shipment_to_db, update_shipment_feedback, get_all_shipments, init_db

# Initialize database
init_db()

app = FastAPI(
    title="FreightSense API",
    description="Backend services for extracting logistics constraints and generating compliance advisories",
    version="1.0.0"
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class FeedbackRequest(BaseModel):
    feedback_notes: str
    corrected_data: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to FreightSense API. Connect to Streamlit dashboard or POST to /extract"}

@app.post("/extract")
async def extract_constraints(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Accepts raw text or an uploaded file (PDF, Image, or Audio note),
    extracts the text, runs NLP Named Entity Recognition and Intent Classification,
    performs RAG against regulations, queries the LLM for compliance risk advisory,
    validates the output, saves the record, and returns the structured results.
    """
    raw_input_text = ""
    
    # 1. Handle Ingestion depending on inputs
    if file:
        file_ext = os.path.splitext(file.filename)[1].lower()
        temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
        
        # Save uploaded file temporarily
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process based on extension
            if file_ext == ".pdf":
                print(f"📥 Processing PDF file: {file.filename}")
                raw_input_text = extract_text_from_pdf(temp_file_path)
            elif file_ext in [".png", ".jpg", ".jpeg"]:
                print(f"📥 Processing Image file: {file.filename}")
                raw_input_text = extract_text_from_image(temp_file_path)
            elif file_ext in [".wav", ".mp3", ".m4a", ".ogg", ".aac"]:
                print(f"📥 Processing Audio file: {file.filename}")
                # Transcribe using local Whisper
                raw_input_text = transcribe_audio(temp_file_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    elif text:
        raw_input_text = text
    else:
        raise HTTPException(status_code=400, detail="Either 'text' form field or 'file' upload must be provided.")

    if not raw_input_text.strip():
        raise HTTPException(status_code=422, detail="No readable text could be extracted from the input.")

    # 2. Run Pipeline Steps
    try:
        # Step 1: Clean text
        cleaned = clean_text(raw_input_text)
        
        # Step 2: Extract entities
        entities = extract_entities(cleaned)
        
        # Step 3: Classify intents
        intents = get_sentence_intents(cleaned)
        
        # Step 4: Retrieve context via RAG
        retrieved_regulations = retrieve_context(cleaned, n_results=3)
        
        # Step 5: Format Prompt
        nlp_results = {
            "text": cleaned,
            "entities": entities,
            "intents": intents
        }
        prompt = build_compliance_prompt(nlp_results, retrieved_regulations)
        
        # Step 6: Call LLM
        raw_llm_response = query_llm_advisory(prompt)
        
        # Step 7: Validate Output JSON
        validated_advisory = parse_and_validate_advisory(raw_llm_response)
        
        # Generate Unique Shipment ID
        shipment_id = f"SHP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        
        # Step 8: Generate Driver WhatsApp Alert
        driver_alert = generate_driver_whatsapp_alert(shipment_id, validated_advisory)
        
        # Save Shipment to Database
        save_shipment_to_db(shipment_id, cleaned, entities, intents, validated_advisory)
        
        return {
            "shipment_id": shipment_id,
            "raw_text": cleaned,
            "entities": entities,
            "intents": intents,
            "advisory": validated_advisory,
            "driver_whatsapp_alert": driver_alert
        }
        
    except Exception as e:
        print(f"❌ Pipeline processing crashed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

@app.get("/shipments")
def list_shipments():
    """
    Returns all processed shipment records logged in the database.
    """
    records = get_all_shipments()
    response = []
    for r in records:
        response.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "raw_input_text": r.raw_input_text,
            "extracted_entities": json.loads(r.extracted_entities) if r.extracted_entities else [],
            "sentence_intents": json.loads(r.sentence_intents) if r.sentence_intents else [],
            "advisory_json": json.loads(r.advisory_json) if r.advisory_json else {},
            "overall_risk": r.overall_risk,
            "has_feedback": r.has_feedback,
            "feedback_notes": r.feedback_notes,
            "corrected_data": json.loads(r.corrected_data) if r.corrected_data else None
        })
    return response

class SendAlertRequest(BaseModel):
    phone_number: str
    alert_type: str = "whatsapp"  # whatsapp or sms

@app.post("/feedback/{shipment_id}")
def submit_feedback(shipment_id: str, request: FeedbackRequest):
    """
    Logs feedback/override notes and corrected JSON data from the dispatcher
    for a specific shipment.
    """
    success = update_shipment_feedback(shipment_id, request.feedback_notes, request.corrected_data)
    if not success:
        raise HTTPException(status_code=404, detail=f"Shipment record {shipment_id} not found.")
    return {"status": "success", "message": "Feedback logged successfully for retraining."}

@app.post("/send_alert/{shipment_id}")
def send_shipment_alert(shipment_id: str, request: SendAlertRequest):
    """
    Sends the compiled driver alert payload to the specified phone number
    or Telegram Chat ID.
    """
    from src.feedback.feedback_db import SessionLocal, ShipmentRecord
    db = SessionLocal()
    try:
        record = db.query(ShipmentRecord).filter(ShipmentRecord.id == shipment_id).first()
        if not record:
            raise HTTPException(status_code=404, detail=f"Shipment record {shipment_id} not found.")
            
        # Re-generate the alert payload from stored advisory JSON
        advisory_dict = json.loads(record.advisory_json) if record.advisory_json else {}
        alert_body = generate_driver_whatsapp_alert(shipment_id, advisory_dict)
        
        # Route depending on the alert type
        alert_type_lower = request.alert_type.lower()
        if alert_type_lower == "whatsapp":
            from src.output.twilio_sender import send_whatsapp_alert
            res = send_whatsapp_alert(request.phone_number, alert_body)
        elif alert_type_lower == "sms":
            from src.output.twilio_sender import send_sms_alert
            res = send_sms_alert(request.phone_number, alert_body)
        elif alert_type_lower == "telegram":
            from src.output.telegram_sender import send_telegram_alert
            res = send_telegram_alert(request.phone_number, alert_body)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported alert type: {request.alert_type}")
            
        return res
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
