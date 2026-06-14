# -*- coding: utf-8 -*-
import pytest
from unittest.mock import patch, MagicMock
from src.genai.agentic_rag import AgenticRAG

@pytest.fixture
def mock_chroma():
    with patch("src.genai.agentic_rag.get_chroma_collection") as mock_get_collection:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {"documents": [["Rule 1: No trucks in Delhi", "Rule 2: Speed limit 40"]]}
        mock_get_collection.return_value = mock_collection
        yield mock_get_collection

@pytest.fixture
def mock_genai():
    with patch("src.genai.agentic_rag.genai.GenerativeModel") as mock_model_class:
        mock_instance = MagicMock()
        mock_model_class.return_value = mock_instance
        yield mock_instance

def test_agentic_rag_high_confidence(mock_chroma, mock_genai):
    """Test that the loop stops after 1 iteration if confidence is high."""
    
    # Mock the LLM returning high confidence
    mock_response = MagicMock()
    mock_response.text = """{
        "overall_risk": "LOW",
        "risk_score": 0.1,
        "advisory_text": "All good.",
        "risk_flags": [],
        "route_recommendations": [],
        "self_critique": {
            "overall_confidence": 0.95
        },
        "iteration": 1
    }"""
    mock_genai.generate_content.return_value = mock_response
    
    rag = AgenticRAG(max_iterations=3, confidence_threshold=0.75)
    rag.model = mock_genai
    
    constraints = {"cargo_type": ["apples"], "locations": ["Delhi"]}
    result = rag.run(constraints)
    
    assert result["total_iterations"] == 1
    assert result["overall_risk"] == "LOW"
    assert len(result["reasoning_chain"]) == 1
    assert result["reasoning_chain"][0]["confidence"] == 0.95

def test_agentic_rag_low_confidence_loop(mock_chroma, mock_genai):
    """Test that the loop runs multiple times if confidence is low."""
    
    # Mock responses: 1st call low confidence, 2nd call high confidence
    response1 = MagicMock()
    response1.text = """{
        "overall_risk": "MEDIUM",
        "risk_score": 0.5,
        "advisory_text": "Not sure about truck timings.",
        "risk_flags": [],
        "route_recommendations": [],
        "self_critique": {
            "overall_confidence": 0.5,
            "follow_up_query": "Delhi truck entry timings"
        },
        "iteration": 1
    }"""
    
    response2 = MagicMock()
    response2.text = """{
        "overall_risk": "HIGH",
        "risk_score": 0.8,
        "advisory_text": "Trucks banned 8AM-8PM.",
        "risk_flags": [],
        "route_recommendations": [],
        "self_critique": {
            "overall_confidence": 0.9
        },
        "iteration": 2
    }"""
    
    mock_genai.generate_content.side_effect = [response1, response2]
    
    rag = AgenticRAG(max_iterations=3, confidence_threshold=0.75)
    rag.model = mock_genai
    
    constraints = {"locations": ["Delhi"]}
    result = rag.run(constraints)
    
    assert result["total_iterations"] == 2
    assert len(result["reasoning_chain"]) == 2
    assert result["reasoning_chain"][0]["confidence"] == 0.5
    assert result["reasoning_chain"][1]["confidence"] == 0.9

def test_agentic_rag_fallback(mock_chroma, mock_genai):
    """Test fallback when LLM fails or returns invalid JSON."""
    
    mock_genai.generate_content.side_effect = Exception("API Error")
    
    rag = AgenticRAG(max_iterations=3)
    rag.model = mock_genai
    
    result = rag.run({"locations": ["Delhi"]})
    
    assert result["overall_risk"] == "MEDIUM"
    assert result["self_critique"]["missing_info"] == "Model failed"
