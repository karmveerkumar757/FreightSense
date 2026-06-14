# -*- coding: utf-8 -*-
"""
Agentic RAG with Self-Critique Loop for FreightSense
Architecture: ReAct pattern (Reason + Act)
- Step 1: Initial retrieval + advisory generation
- Step 2: Self-critique (confidence scoring on 4 dimensions)  
- Step 3: If confidence < threshold, formulate new retrieval query
- Step 4: Re-retrieve + regenerate (max 3 iterations)
- Step 5: Return final advisory with full reasoning chain
"""
import os
import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from typing import List, Dict, Any
from src.genai.rag_retriever import hybrid_rerank, get_chroma_collection

class AgenticRAG:
    def __init__(self, max_iterations: int = 3, confidence_threshold: float = 0.75):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            # Use gemini-2.5-flash as requested
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            print("⚠️ GEMINI_API_KEY not found in environment.")
            self.model = None

    def retrieve(self, query: str, n_results: int = 5) -> List[str]:
        """
        Retrieves top relevant chunks using Hybrid Search (Semantic + Keyword TF-IDF).
        """
        collection = get_chroma_collection()
        count = collection.count()
        if count == 0:
            return []
            
        semantic_fetch_count = min(15, count)
        results = collection.query(
            query_texts=[query],
            n_results=semantic_fetch_count
        )
        
        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            retrieved_docs = results["documents"][0]
            
        if retrieved_docs:
            retrieved_docs = hybrid_rerank(query, retrieved_docs, top_k=n_results)
            
        return retrieved_docs

    def generate_advisory(self, constraints: dict, regulation_chunks: List[str], iteration: int) -> dict:
        """
        Calls Gemini with the strict ReAct prompt to generate the advisory AND a self-critique.
        """
        if not self.model:
            return self._get_fallback_response()
            
        reg_context = "\n\n".join([f"CHUNK {i+1}:\n{chunk}" for i, chunk in enumerate(regulation_chunks)])
        if not reg_context:
            reg_context = "No specific local regulations found in database. Use general Indian motor vehicle acts."
            
        constraints_str = json.dumps(constraints, indent=2)
        
        system_prompt = f"""You are a senior Indian logistics compliance officer.

SHIPMENT CONSTRAINTS:
{constraints_str}

RETRIEVED REGULATIONS (iteration {iteration}):
{reg_context}

Generate a risk advisory. You MUST return valid JSON with this exact schema:
{{
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "risk_score": 0.0-1.0,
  "advisory_text": "3 sentence plain English advisory for dispatcher",
  "risk_flags": [
    {{"flag": "description", "severity": "LOW|MEDIUM|HIGH", "regulation_ref": "law name"}}
  ],
  "route_recommendations": ["recommendation 1", "recommendation 2"],
  "self_critique": {{
    "factual_confidence": 0.0-1.0,
    "coverage_confidence": 0.0-1.0,
    "missing_info": "what additional regulations would improve this advisory",
    "follow_up_query": "ChromaDB query to retrieve missing information",
    "overall_confidence": 0.0-1.0
  }},
  "iteration": {iteration}
}}
Return ONLY the JSON. No markdown wrappers. No explanation outside JSON.
"""
        try:
            response = self.model.generate_content(system_prompt)
            raw_text = response.text.strip()
            
            # Clean markdown code blocks if the model ignored instructions
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            return json.loads(raw_text.strip())
        except Exception as e:
            print(f"⚠️ AgenticRAG generation failed: {e}")
            return self._get_fallback_response()

    def _build_initial_query(self, constraints: dict) -> str:
        """
        Builds the starting search query based on the extracted entities.
        """
        query_parts = []
        if constraints.get("cargo_type"):
            query_parts.extend(constraints["cargo_type"])
        if constraints.get("locations"):
            query_parts.extend(constraints["locations"])
        if constraints.get("compliance_reqs"):
            query_parts.extend(constraints["compliance_reqs"])
        if constraints.get("vehicle_type"):
            query_parts.extend(constraints["vehicle_type"])
            
        if not query_parts:
            return "Indian road transport compliance rules"
            
        return " ".join(query_parts)

    def _get_fallback_response(self) -> dict:
        return {
            "overall_risk": "MEDIUM",
            "risk_score": 0.5,
            "advisory_text": "Fallback: Unable to generate detailed advisory. Please manually verify route.",
            "risk_flags": [],
            "route_recommendations": [],
            "self_critique": {
                "factual_confidence": 0.0,
                "coverage_confidence": 0.0,
                "missing_info": "Model failed",
                "follow_up_query": "",
                "overall_confidence": 1.0 # Force break
            },
            "iteration": 1
        }

    def run(self, constraints: dict) -> dict:
        """
        Main ReAct loop execution.
        """
        reasoning_chain = []
        query = self._build_initial_query(constraints)
        
        result = None
        for iteration in range(1, self.max_iterations + 1):
            print(f"🔄 Agentic RAG Iteration {iteration}: Searching for '{query}'")
            chunks = self.retrieve(query)
            
            result = self.generate_advisory(constraints, chunks, iteration)
            
            confidence = result.get("self_critique", {}).get("overall_confidence", 1.0)
            reasoning_chain.append({
                "iteration": iteration,
                "query_used": query,
                "chunks_retrieved": len(chunks),
                "confidence": confidence,
                "advisory": result.get("advisory_text", "")
            })
            
            if confidence >= self.confidence_threshold:
                print(f"✅ ReAct Loop Complete: Confidence {confidence} >= {self.confidence_threshold}")
                break
                
            query = result.get("self_critique", {}).get("follow_up_query", "")
            if not query:
                break
                
        if result:
            result["reasoning_chain"] = reasoning_chain
            result["total_iterations"] = iteration
            return result
        return self._get_fallback_response()
