import os
import requests
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

sample_instruction = "Urgent: Deliver pharmaceutical goods to Apollo Hospital Delhi before 4 PM. Avoid Outer Ring Road due to heavy traffic."

def test_ollama():
    print("\n🤖 Testing Local Ollama (Llama 3)...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": f"Extract logistics constraints from this note: {sample_instruction}",
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            print("✅ Ollama Success Response:")
            print(response.json().get("response"))
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Ollama connection failed. Is the Ollama app running? Error: {e}")

def test_gemini():
    print("\n☁️ Testing Google Gemini API...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file.")
        return
        
    import time
    import random
    
    models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-3.5-flash']
    client = genai.Client(api_key=api_key)
    
    for model in models_to_try:
        print(f"👉 Trying model: {model}...")
        max_retries = 3
        delay = 1.0
        success = False
        
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"Summarize this logistics note in one sentence: {sample_instruction}"
                )
                print(f"✅ Gemini Success Response (Model: {model}):")
                print(response.text.strip())
                success = True
                break
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(code in err_str for code in ["503", "429", "unavailable", "resource_exhausted", "high demand", "too many requests"])
                
                if is_transient and attempt < max_retries:
                    sleep_time = delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    print(f"  ⏳ Transient error on {model} (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"  ❌ Model {model} failed on attempt {attempt}: {e}")
                    break
        if success:
            return
            
    print("❌ All Gemini models failed to respond.")

if __name__ == "__main__":
    # Ensure Ollama is running locally on your system before execution!
    test_ollama()
    test_gemini()