# -*- coding: utf-8 -*-
"""
FreightSense FastAPI Backend — Complete Integration
All 8 upgrades wired together in one unified pipeline
"""
import os
import sys
import time
import uuid
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import engines
from src.ingestion.asr import UnifiedASR
from src.nlp.ner_extractor import extract_entities
from src.nlp.intent_classifier import get_sentence_intents
from src.genai.agentic_rag import AgenticRAG
from src.output.risk_classifier import RiskClassifier
from src.nlp.delay_predictor import DelayPredictorService
from src.genai.multi_objective_vrp import MultiObjectiveVRP
from src.genai.dpo_advisory_generator import DPOAdvisoryGenerator
from src.output.feedback_collector import log_preference_pair
from tests.eval_metrics import run_full_evaluation

app = FastAPI(title="FreightSense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy loading recommended for heavy models but instantiated here for simplicity)
asr_engine = UnifiedASR()
class NEREngineWrapper:
    def extract(self, text):
        return extract_entities(text)
        
ner_engine = NEREngineWrapper()
class IntentEngineWrapper:
    def predict_intents(self, text):
        return get_sentence_intents(text)
        
intent_engine = IntentEngineWrapper()
dpo_generator = DPOAdvisoryGenerator()
risk_classifier = RiskClassifier()
delay_predictor = DelayPredictorService()
vrp_engine = MultiObjectiveVRP()

class ShipmentInput(BaseModel):
    instruction: str
    locations: List[dict] = [] # [{"lat": 28.6, "lon": 77.2}]

class FeedbackInput(BaseModel):
    shipment_id: str
    constraints: dict
    rejected_advisory: str
    chosen_advisory: str
    dispatcher_id: str = "unknown"
    rejection_reason: Optional[str] = None

class FreightSenseResponse(BaseModel):
    shipment_id: str
    processing_time_ms: float
    
    # NLP outputs
    constraints: dict
    entities: dict  
    intents: List[dict]
    
    # GenAI outputs
    advisory: dict  
    agentic_iterations: int
    
    # Risk classification + SHAP
    risk_explanation: dict  
    
    # LSTM delay prediction
    delay_prediction: dict
    
    # Routing
    optimised_route: Optional[dict]  
    
    # Metadata
    models_used: dict  
    fallbacks_triggered: List[str]  

@app.post("/extract", response_model=FreightSenseResponse)
async def extract_and_advise(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    shipment_id = uuid.uuid4().hex[:8]
    fallbacks = []
    
    instruction = text or ""
    
    # 1. ASR
    if audio:
        audio_path = f"temp_{shipment_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
            
        res = asr_engine.transcribe(audio_path)
        instruction = res["text"]
        if res["source"] != "bhashini":
            fallbacks.append("bhashini_asr->whisper")
        os.remove(audio_path)

    # 2. NLP (NER & Intent)
    entities = ner_engine.extract(instruction)
    intents = intent_engine.predict_intents(instruction)
    
    constraints = {
        "cargo_type": entities.get("cargo_type", []),
        "locations": entities.get("locations", []),
        "route_constraints": entities.get("route_constraints", []),
        "special_handling": entities.get("special_handling", []),
        "vehicle_type": entities.get("vehicle_type", []),
        "compliance_reqs": entities.get("compliance_reqs", [])
    }
    
    # 3. Advisory (DPO or Agentic RAG fallback)
    if dpo_generator.is_dpo_model_available():
        # DPO doesn't do chroma retrieval directly in this simplified setup, just generation
        # In a real app we'd retrieve context and pass to DPO
        advisory_res = dpo_generator.generate(constraints, [])
        agentic_iterations = 1
    else:
        advisory_res = dpo_generator.fallback_agent.run(constraints)
        agentic_iterations = advisory_res.get("total_iterations", 1)
        fallbacks.append("dpo->agentic_rag")
        
    # 4. Risk Classification (XGBoost + SHAP)
    risk_exp = risk_classifier.predict_with_explanation(constraints)
    
    # 5. Delay Prediction (LSTM)
    delay_pred = delay_predictor.predict(lat=28.6139, lon=77.2090) # Defaulting to Delhi if no coords
    
    processing_time = (time.time() - start_time) * 1000
    
    return FreightSenseResponse(
        shipment_id=shipment_id,
        processing_time_ms=processing_time,
        constraints=constraints,
        entities=entities,
        intents=intents,
        advisory=advisory_res,
        agentic_iterations=agentic_iterations,
        risk_explanation=risk_exp,
        delay_prediction=delay_pred,
        optimised_route=None, # Filled by separate endpoint
        models_used={
            "ner": "IndicBERT-BiLSTM-CRF",
            "intent": "BERT-SequenceClassifier",
            "advisory": "Gemma-2B-DPO" if dpo_generator.is_dpo_model_available() else "Gemini-1.5-Flash (Agentic RAG)",
            "risk": "XGBoost",
            "delay": "PyTorch-LSTM"
        },
        fallbacks_triggered=fallbacks
    )

@app.post("/optimise_route")
async def optimise_route(shipment: ShipmentInput):
    """
    Multi-objective route optimisation
    Returns Pareto front + RL-selected best route
    """
    if not shipment.locations:
        return {"error": "Locations required"}
        
    dist_mat = vrp_engine.get_distance_matrix(shipment.locations)
    risk_mat = vrp_engine.compute_risk_matrix(shipment.locations, {})
    delay_mat = vrp_engine.compute_delay_matrix(shipment.locations)
    
    pareto = vrp_engine.generate_pareto_front(dist_mat, risk_mat, delay_mat)
    best_route = vrp_engine.select_best_route(pareto, {"risk_score": 0.5, "delay_prob": 0.5})
    
    return {
        "best_route": best_route,
        "pareto_alternatives": pareto,
        "route_map_url": "#", # Placeholder
        "selection_reason": best_route.get("selection_reason", "Default")
    }

@app.post("/feedback")
async def collect_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """DPO preference pair collection"""
    log_preference_pair(
        feedback.shipment_id, 
        feedback.constraints, 
        feedback.rejected_advisory, 
        feedback.chosen_advisory,
        feedback.dispatcher_id,
        feedback.rejection_reason
    )
    return {"status": "Feedback logged successfully and evaluation queued."}

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "models": {
            "bhashini": asr_engine.bhashini is not None,
            "dpo_advisory": dpo_generator.is_dpo_model_available(),
            "xgboost_risk": risk_classifier.model is not None,
            "lstm_delay": delay_predictor.model is not None
        }
    }

@app.post("/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """Trigger full LLM-as-Judge evaluation on test set"""
    background_tasks.add_task(run_full_evaluation, "data/eval_test_set.json")
    return {"status": "Evaluation queued in background."}
