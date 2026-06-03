# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TimeWindow(BaseModel):
    earliest: Optional[str] = Field(default=None, description="Earliest delivery time (e.g. '10:00')")
    latest: Optional[str] = Field(default=None, description="Latest delivery time (e.g. '18:00')")

class CargoMetadata(BaseModel):
    type: str = Field(description="Cargo type name")
    handling: List[str] = Field(default_factory=list, description="List of handling requirements")

class ComplianceMetadata(BaseModel):
    eway_bill_valid_until: Optional[str] = Field(default=None, description="e-Way Bill expiration ISO timestamp or description")
    dangerous_goods: bool = Field(default=False, description="Flag for hazardous materials")
    permits_required: List[str] = Field(default_factory=list, description="Required state permits or transshipment copies")

class RiskItem(BaseModel):
    type: str = Field(description="Type of risk (e.g. route_risk, time_risk, compliance_risk)")
    severity: str = Field(description="Severity levels: high, medium, low")
    description: str = Field(description="Explanation of the risk condition")

class ShipmentComplianceSchema(BaseModel):
    deadline: Optional[str] = Field(default=None, description="Target arrival deadline")
    time_window: TimeWindow = Field(default_factory=TimeWindow)
    priority: str = Field(default="medium", description="Priority ranking")
    cargo: CargoMetadata
    vehicle_required: Optional[str] = Field(default=None, description="Recommended vehicle model or capacity")
    avoid_routes: List[str] = Field(default_factory=list, description="Highways or road structures to avoid")
    compliance: ComplianceMetadata = Field(default_factory=ComplianceMetadata)
    risks: List[RiskItem] = Field(default_factory=list)
    overall_risk: str = Field(default="low", description="Advisory danger rank: low, medium, high")
    confidence: float = Field(default=1.0, description="LLM logic confidence level")
    plain_english_advisory: str = Field(description="Plain English summary for dispatcher")
    driver_alerts: List[str] = Field(default_factory=list, description="List of driver alert lines")

def clean_llm_json_string(raw_response: str) -> str:
    """
    Cleans markdown code fences and whitespace from LLM json output
    to extract a raw valid JSON string.
    """
    cleaned = raw_response.strip()
    
    # Strip markdown block codes if LLM returned them
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
        
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
        
    return cleaned.strip()

def parse_and_validate_advisory(raw_llm_output: str) -> Dict[str, Any]:
    """
    Parses LLM response, cleans any markdown formatting, and validates
    it against the Pydantic schema to ensure all keys are present.
    """
    json_str = clean_llm_json_string(raw_llm_output)
    
    try:
        parsed_dict = json.loads(json_str)
        # Validate using Pydantic model
        validated_model = ShipmentComplianceSchema(**parsed_dict)
        return validated_model.model_dump()
    except Exception as e:
        print(f"⚠️ Pydantic Validation failed: {e}. Attempting manual parsing correction...")
        # Fallback parsing strategy in case validation failed completely
        try:
            parsed_dict = json.loads(json_str)
            # Ensure essential keys are present
            if "cargo" not in parsed_dict:
                parsed_dict["cargo"] = {"type": "unknown", "handling": []}
            if "compliance" not in parsed_dict:
                parsed_dict["compliance"] = {"dangerous_goods": False, "permits_required": []}
            if "risks" not in parsed_dict:
                parsed_dict["risks"] = []
            if "overall_risk" not in parsed_dict:
                parsed_dict["overall_risk"] = "low"
            if "plain_english_advisory" not in parsed_dict:
                parsed_dict["plain_english_advisory"] = "Shipment parsed with validation error. Please review manually."
            if "driver_alerts" not in parsed_dict:
                parsed_dict["driver_alerts"] = ["Check shipment constraints manually."]
            return parsed_dict
        except Exception as json_err:
            raise ValueError(f"CRITICAL: Failed to parse LLM response as JSON: {json_err}. Raw output was: {raw_llm_output}")
