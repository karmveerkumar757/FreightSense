import sys
import traceback

modules_to_test = [
    "src.api.main",
    "app",
    "src.ingestion.asr",
    "src.ingestion.bhashini_asr",
    "src.ingestion.ocr",
    "src.nlp.bilstm_crf_ner",
    "src.nlp.intent_classifier",
    "src.nlp.delay_predictor",
    "src.genai.agentic_rag",
    "src.genai.dpo_advisory_generator",
    "src.genai.multi_objective_vrp",
    "src.output.risk_classifier"
]

for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"✅ {mod} imported successfully.")
    except Exception as e:
        print(f"❌ {mod} failed: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stdout)
