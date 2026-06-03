# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re

def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw text input (OCR / ASR transcription / direct input).
    """
    if not text:
        return ""
        
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespaces
    text = text.strip()
    
    return text
