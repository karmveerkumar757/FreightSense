import sys
import traceback

def check_imports():
    modules = [
        "src.api.main",
        "app",
        "src.genai.agentic_rag",
        "src.output.risk_classifier",
        "src.nlp.delay_predictor",
        "src.genai.multi_objective_vrp",
        "src.genai.rl_weight_agent",
        "src.output.feedback_collector",
        "src.genai.dpo_advisory_generator",
        "src.ingestion.bhashini_asr",
        "src.output.bhashini_tts",
        "src.ingestion.asr",
        "tests.llm_judge",
        "tests.eval_metrics"
    ]
    
    errors = []
    for mod in modules:
        try:
            __import__(mod)
            print(f"✅ Successfully imported {mod}")
        except Exception as e:
            print(f"❌ Failed to import {mod}: {e}")
            errors.append((mod, traceback.format_exc()))
            
    if errors:
        print("\n\n--- Error Details ---")
        for mod, tb in errors:
            print(f"=== {mod} ===")
            print(tb)

if __name__ == "__main__":
    check_imports()
