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
    # Format the regulatory context
    regulations_context = ""
    if retrieved_regulations:
        for idx, chunk in enumerate(retrieved_regulations):
            regulations_context += f"Regulation Chunk #{idx+1}:\n{chunk}\n\n"
    else:
        regulations_context = "No specific local regulatory document matches found in database. Use general Indian road safety and national transport rules."

    # Convert the extracted NLP details to pretty JSON string
    nlp_formatted = json.dumps(extracted_nlp, indent=2)

    prompt = f"""
SYSTEM: You are a senior logistics compliance officer and route planner for Indian freight shipping operations.
Your job is to read unstructured shipment text/notes, review the extracted entities and intents, check them against the relevant Indian transport regulations, and output a structured risk advisory.

EXTRACTED SHIPMENT DETAILS:
{nlp_formatted}

RELEVANT LOGISTICS REGULATIONS (Retrieved Context):
{regulations_context}

YOUR CORE TASKS:
1. Review the cargo, deadline, route constraints, special handling, and compliance requirements against the regulations.
2. Identify all potential legal or physical violations, risk flags, or conflicts (e.g. e-way bill validity issues, state truck entry restrictions, hazardous materials guidelines, dimensions rules, overloading under the MV Act 2019).
3. Recommend specific route adjustments or alternative actions (e.g. use bypass roads, request e-way bill extensions, avoid specific tollways).
4. Assign an overall risk rating: "low", "medium", or "high".
5. Estimate the confidence level (a float between 0.0 and 1.0) of your analysis.
6. Generate a 3-sentence plain-English advisory summary for dispatchers.
7. Generate a WhatsApp-friendly short alert list for the driver.

STRICT OUTPUT FORMAT:
You MUST output your response as a valid JSON object ONLY. Do not include markdown code fence wrappers (like ```json ... ```) or any trailing explanations. Start with '{{' and end with '}}'.

JSON SCHEMA REQUIRED:
{{
  "deadline": "string or null (ISO datetime format or descriptive time)",
  "time_window": {{
    "earliest": "string or null (e.g., '10:00')",
    "latest": "string or null (e.g., '18:00')"
  }},
  "priority": "string (e.g., 'high', 'medium', 'low')",
  "cargo": {{
    "type": "string (e.g., 'pharmaceutical', 'electronics')",
    "handling": ["list of strings (e.g. 'refrigerated', 'fragile')"]
  }},
  "vehicle_required": "string or null (e.g. 'refrigerated_truck_min_15ft')",
  "avoid_routes": ["list of strings (e.g. 'Ring_Road_Delhi')"],
  "compliance": {{
    "eway_bill_valid_until": "string or null",
    "dangerous_goods": false,
    "permits_required": ["list of strings"]
  }},
  "risks": [
    {{
      "type": "string (e.g., 'time_risk', 'route_risk', 'compliance_risk')",
      "severity": "string (e.g. 'high', 'medium', 'low')",
      "description": "string (reasoning behind this risk)"
    }}
  ],
  "overall_risk": "string ('low', 'medium', or 'high')",
  "confidence": 0.95,
  "plain_english_advisory": "string (exactly 3 sentences summarizing risk/routing advice for dispatchers)",
  "driver_alerts": [
    "string (alert 1)",
    "string (alert 2)"
  ]
}}
"""
    return prompt
