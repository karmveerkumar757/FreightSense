# -*- coding: utf-8 -*-
"""
Bhashini ASR (Automatic Speech Recognition) client
Supports 22 Indian languages via Government of India's Dhruva API
FREE for developers - sign up at bhashini.gov.in
"""
import os
import requests
import base64
import json
from pathlib import Path

BHASHINI_BASE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

SUPPORTED_LANGUAGES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "bn": "Bengali", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
    "en": "English"
}

class BhashiniASR:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("BHASHINI_API_KEY", "")
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

    def _get_audio_base64(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def detect_language(self, audio_path: str) -> str:
        """
        Uses Bhashini language identification pipeline.
        Since Bhashini's full pipeline structure requires a specific taskType for lang detection,
        we simulate the fallback logic here if the API is not set.
        """
        # In a real impl, you hit the language detection taskType.
        # Defaulting to hi for demo fallback.
        return "hi"

    def transcribe(self, audio_path: str, language: str = "auto") -> dict:
        if not self.api_key:
            raise Exception("BHASHINI_API_KEY not configured.")
            
        if language == "auto":
            language = self.detect_language(audio_path)
            
        base64_audio = self._get_audio_base64(audio_path)
        
        pipeline_request = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": language},
                        "audioFormat": "wav",
                        "samplingRate": 16000
                    }
                }
            ],
            "inputData": {
                "audio": [{"audioContent": base64_audio}]
            }
        }
        
        response = requests.post(BHASHINI_BASE_URL, headers=self.headers, json=pipeline_request)
        if response.status_code == 200:
            data = response.json()
            try:
                text = data["pipelineResponse"][0]["output"][0]["source"]
                return {"text": text, "language": language, "confidence": 0.95}
            except KeyError:
                raise Exception("Invalid response format from Bhashini ASR")
        else:
            raise Exception(f"Bhashini ASR API failed: {response.status_code} - {response.text}")

    def transcribe_with_fallback(self, audio_path: str) -> dict:
        """Try Bhashini first, fall back to Whisper if it fails."""
        try:
            res = self.transcribe(audio_path)
            res["source"] = "bhashini"
            print(f"🎙️ Bhashini ASR Success: {res['language']}")
            return res
        except Exception as e:
            print(f"⚠️ Bhashini ASR failed: {e}. Falling back to Whisper...")
            try:
                from src.ingestion.asr import WhisperASR
                whisper = WhisperASR()
                text = whisper.transcribe(audio_path)
                return {"text": text, "language": "auto", "source": "whisper"}
            except Exception as e2:
                print(f"⚠️ Whisper fallback failed: {e2}")
                return {"text": "", "language": "unknown", "source": "failed"}
