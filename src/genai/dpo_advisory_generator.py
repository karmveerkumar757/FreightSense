# -*- coding: utf-8 -*-
"""
Advisory generator that uses DPO fine-tuned Gemma-2B if available,
falls back to Gemini API if not (handles both deployment scenarios)
"""
import os
import json
import torch
from typing import List

class DPOAdvisoryGenerator:
    def __init__(self):
        self.adapter_path = os.path.join("models", "gemma_dpo_adapter")
        self.model_id = "google/gemma-2b-it"
        self.model = None
        self.tokenizer = None
        self.use_dpo = self._load_model()
        
        if not self.use_dpo:
            # Fallback to Agentic RAG / Gemini
            from src.genai.agentic_rag import AgenticRAG
            self.fallback_agent = AgenticRAG()

    def _load_model(self) -> bool:
        """Check if DPO adapter exists and load it"""
        if os.path.exists(self.adapter_path):
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from peft import PeftModel
                
                print(f"🔄 Loading base model {self.model_id}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                print(f"🔄 Applying DPO adapter from {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                return True
            except ImportError:
                print("⚠️ Transformers/PEFT not installed. Falling back to Gemini.")
            except Exception as e:
                print(f"⚠️ Failed to load DPO model: {e}")
                
        return False

    def is_dpo_model_available(self) -> bool:
        """Check if fine-tuned model exists"""
        return self.use_dpo

    def generate(self, constraints: dict, regulation_chunks: List[str]) -> dict:
        """Generate advisory using whichever model is available"""
        if self.use_dpo and self.model and self.tokenizer:
            print("🚀 Generating advisory using local DPO-tuned Gemma-2B...")
            
            prompt = f"Generate a risk advisory for this freight: {json.dumps(constraints)}\n\n"
            if regulation_chunks:
                prompt += "Context:\n" + "\n".join(regulation_chunks)
                
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
                
            response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            return {
                "advisory_text": response_text.strip(),
                "source": "gemma_2b_dpo",
                "overall_risk": "UNKNOWN" # Local model might not generate perfect JSON, we extract text
            }
        else:
            print("☁️ Generating advisory using Gemini API (Agentic RAG Fallback)...")
            return self.fallback_agent.run(constraints)
