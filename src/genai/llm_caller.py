# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import time
import random
import requests
from dotenv import load_dotenv
from google import genai
from typing import Dict, Any

load_dotenv()

def call_gemini(prompt: str, model_name: str = 'gemini-2.5-flash', max_retries: int = 5, initial_delay: float = 1.0) -> str:
    """
    Calls Google Gemini API using the new modern SDK with exponential backoff and jitter
    for transient errors (like 503 unavailable or 429 resource exhausted).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file.")
        
    client = genai.Client(api_key=api_key)
    
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=dict(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(code in err_str for code in ["503", "429", "unavailable", "resource_exhausted", "high demand", "too many requests"])
            
            if is_transient and attempt < max_retries:
                sleep_time = delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                print(f"⏳ Gemini API returned transient error (attempt {attempt}/{max_retries}) for model {model_name}: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                raise e

def call_ollama(prompt: str, model_name="llama3") -> str:
    """
    Calls a local Ollama server instance (e.g., Llama 3 or 3.1).
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": "json",  # Instruct Ollama to output valid JSON
        "stream": False
    }
    
    response = requests.post(url, json=payload, timeout=90)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise RuntimeError(f"Ollama server returned status code: {response.status_code}")

def query_llm_advisory(prompt: str) -> str:
    """
    Queries the LLM for compliance risk advisory.
    Prioritizes Gemini 2.5-flash, falls back to Gemini 2.0-flash, then Gemini 3.5-flash, and finally local Ollama.
    """
    # 1. Try Gemini 2.5-flash
    if os.getenv("GEMINI_API_KEY"):
        try:
            print("☁️ Querying Google Gemini (gemini-2.5-flash)...")
            return call_gemini(prompt, 'gemini-2.5-flash')
        except Exception as e:
            print(f"⚠️ Gemini 2.5 API failed: {e}. Trying Gemini 2.0-flash...")
            
        # 2. Try Gemini 2.0-flash
        try:
            print("☁️ Querying Google Gemini (gemini-2.0-flash)...")
            return call_gemini(prompt, 'gemini-2.0-flash')
        except Exception as e:
            print(f"⚠️ Gemini 2.0 API failed: {e}. Trying Gemini 3.5-flash...")

        # 3. Try Gemini 3.5-flash
        try:
            print("☁️ Querying Google Gemini (gemini-3.5-flash)...")
            return call_gemini(prompt, 'gemini-3.5-flash')
        except Exception as e:
            print(f"⚠️ Gemini 3.5 API failed: {e}. Falling back to local Ollama...")
            
    # 4. Try Local Ollama
    try:
        print("🤖 Querying Local Ollama (llama3)...")
        return call_ollama(prompt)
    except Exception as e:
        print(f"❌ Local Ollama query failed: {e}")
        
    raise RuntimeError("All configured Gemini and Ollama models failed to respond.")
