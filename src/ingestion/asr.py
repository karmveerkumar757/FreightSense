# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper ASR not installed. Voice transcription will fallback to Bhashini or text only.")

_model = None

def get_whisper_model(model_name="tiny"):
    """
    Loads and caches the Whisper model to avoid reloading on every request.
    Defaults to 'tiny' for fast execution and low memory consumption.
    """
    global _model
    if _model is None:
        print(f"📦 Loading local Whisper model '{model_name}' (this may take a moment on first run)...")
        # We can run on GPU if available, otherwise CPU
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = whisper.load_model(model_name, device=device)
        print("✅ Whisper model loaded successfully.")
    return _model

class WhisperASR:
    def __init__(self, model_name="tiny"):
        if not WHISPER_AVAILABLE:
            self.model = None
            return
            
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes audio from the given file path to text using Whisper.
        Returns the transcribed text.
        """
        print(f"🎙️ WhisperASR: Processing {audio_path}...")
        if not self.model:
            print("❌ Whisper model not available (not installed).")
            return ""
            
        try:
            result = self.model.transcribe(audio_path, fp16=self.device.type == "cuda")
            text = result["text"].strip()
            print("✅ Transcription complete.")
            return text
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""

class UnifiedASR:
    """
    Tries Bhashini first (better Indian language support)
    Falls back to Whisper (better for code-switching Hindi-English)
    Falls back to raw text if both fail
    """
    def __init__(self):
        try:
            from src.ingestion.bhashini_asr import BhashiniASR
            self.bhashini = BhashiniASR()
        except ImportError:
            self.bhashini = None
            
        self.whisper = WhisperASR()
    
    def transcribe(self, audio_path: str) -> dict:
        """
        Returns unified format: {"text": str, "language": str, "source": "bhashini|whisper"}
        """
        if self.bhashini and self.bhashini.api_key:
            res = self.bhashini.transcribe_with_fallback(audio_path)
            if res["text"]:
                return res
                
        # If Bhashini fails or isn't loaded, use Whisper
        print("🎙️ Using Whisper ASR...")
        text = self.whisper.transcribe(audio_path)
        return {
            "text": text,
            "language": "auto",
            "source": "whisper"
        }

if __name__ == "__main__":
    pass
