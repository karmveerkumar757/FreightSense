# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
from typing import List, Dict, Any

def build_compliance_prompt(extracted_nlp: Dict[str, Any], retrieved_regulations: List[str]) -> str:
    """
    Constructs the final system prompt to be passed to the LLM (Gemini or Ollama)
    for compliance evaluation, risk assessment, and routing advisory.
    """
    # Note: For V5.0 Agentic RAG, the prompt building is handled internally by agentic_rag.py.
    # This function is kept for backward compatibility with older endpoints.
    
    reg_context = ""
    if retrieved_regulations:
        for idx, chunk in enumerate(retrieved_regulations):
            reg_context += f"Regulation Chunk #{idx+1}:\n{chunk}\n\n"
    else:
        reg_context = "No specific local regulatory document matches found in database. Use general Indian road safety and national transport rules."

    nlp_formatted = json.dumps(extracted_nlp, indent=2)

    prompt = f"""
SYSTEM: You are a senior logistics compliance officer.
Review the constraints against the regulations and output a risk advisory in JSON format.

EXTRACTED SHIPMENT DETAILS:
{nlp_formatted}

RELEVANT LOGISTICS REGULATIONS:
{reg_context}

Return a valid JSON object matching the standard FreightSense advisory schema.
"""
    return prompt
