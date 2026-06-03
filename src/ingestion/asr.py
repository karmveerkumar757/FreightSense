# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import whisper

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

def transcribe_audio(audio_path: str, model_name="tiny") -> str:
    """
    Transcribes an audio file (wav, mp3, m4a, ogg, etc.) using local Whisper.
    Supports mixed English/Hindi audio notes.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    model = get_whisper_model(model_name)
    print(f"🎙️ Transcribing audio file: {audio_path}...")
    
    # We specify task="transcribe" to get original language text or mixed language
    result = model.transcribe(audio_path, task="transcribe")
    text = result.get("text", "")
    return text.strip()
