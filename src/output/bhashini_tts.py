# -*- coding: utf-8 -*-
"""
Bhashini TTS (Text-to-Speech)
Speaks advisory back to driver in their native language
Converts English advisory to target language + generates audio
"""
import os
import requests
import base64
import tempfile
import uuid

BHASHINI_BASE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

class BhashiniTTS:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("BHASHINI_API_KEY", "")
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

    def translate_and_speak(self, english_text: str, target_language: str) -> bytes:
        """
        Two-stage pipeline: En -> Target Lang -> Audio
        """
        if not self.api_key:
            print("[WARN] BHASHINI_API_KEY not configured. Falling back to gTTS.")
            return self._gtts_fallback(english_text, target_language)
            
        if target_language == "en":
            # Just TTS
            pipeline_request = {
                "pipelineTasks": [
                    {
                        "taskType": "tts",
                        "config": {
                            "language": {"sourceLanguage": "en"},
                            "gender": "male",
                            "samplingRate": 8000
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": english_text}]
                }
            }
        else:
            # Translate then TTS
            pipeline_request = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "language": {
                                "sourceLanguage": "en",
                                "targetLanguage": target_language
                            }
                        }
                    },
                    {
                        "taskType": "tts",
                        "config": {
                            "language": {"sourceLanguage": target_language},
                            "gender": "male",
                            "samplingRate": 8000
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": english_text}]
                }
            }

        print(f"[INFO] Requesting Bhashini TTS (en -> {target_language})...")
        response = requests.post(BHASHINI_BASE_URL, headers=self.headers, json=pipeline_request)
        
        if response.status_code == 200:
            data = response.json()
            try:
                # The output format depends on the pipeline, usually the last task's output
                audio_base64 = data["pipelineResponse"][-1]["audio"][0]["audioContent"]
                return base64.b64decode(audio_base64)
            except (KeyError, IndexError) as e:
                print(f"[ERROR] Bhashini TTS format error: {e}. Falling back to gTTS.")
                return self._gtts_fallback(english_text, target_language)
        else:
            print(f"[ERROR] Bhashini TTS API failed: {response.status_code} - {response.text}. Falling back to gTTS.")
            return self._gtts_fallback(english_text, target_language)

    def _gtts_fallback(self, english_text: str, target_language: str) -> bytes:
        try:
            from gtts import gTTS
            from io import BytesIO
            import google.generativeai as genai
            import os
            from dotenv import load_dotenv
            
            load_dotenv() # Load environment variables from .env
            
            text_to_speak = english_text
            if target_language != "en":
                # Translate with Gemini
                api_key = os.getenv("GEMINI_API_KEY", "")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"Translate the following text to language code '{target_language}'. Only return the translated text without any quotes or formatting:\n\n{english_text}"
                    response = model.generate_content(prompt)
                    text_to_speak = response.text.strip()
                    # print removed to avoid UnicodeEncodeError on Windows terminals
            
            tts = gTTS(text=text_to_speak, lang=target_language)
            fp = BytesIO()
            tts.write_to_fp(fp)
            print("[INFO] gTTS fallback generated audio successfully.")
            return fp.getvalue()
        except Exception as e:
            print(f"[ERROR] gTTS fallback failed: {e}")
            return b""

    def save_to_temp_file(self, audio_bytes: bytes) -> str:
        """Saves audio bytes to a temporary wav file."""
        if not audio_bytes:
            return ""
            
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"advisory_{uuid.uuid4().hex[:8]}.wav")
        
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
            
        return file_path

    def send_audio_alert_telegram(self, advisory_text: str, driver_language: str, telegram_chat_id: str):
        """
        Translates text, generates audio, and sends it via Telegram.
        """
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not bot_token or not telegram_chat_id:
            print("[WARN] TELEGRAM_BOT_TOKEN or chat_id missing. Cannot send audio.")
            return
            
        audio_bytes = self.translate_and_speak(advisory_text, driver_language)
        if not audio_bytes:
            return
            
        file_path = self.save_to_temp_file(audio_bytes)
        if not file_path:
            return
            
        url = f"https://api.telegram.org/bot{bot_token}/sendVoice"
        
        with open(file_path, "rb") as f:
            files = {"voice": f}
            data = {"chat_id": telegram_chat_id, "caption": "Risk Advisory Audio (Bhashini TTS)"}
            res = requests.post(url, data=data, files=files)
            
        if res.status_code == 200:
            print(f"[SUCCESS] Audio advisory sent successfully to Telegram ({telegram_chat_id})")
        else:
            print(f"[ERROR] Failed to send audio to Telegram: {res.text}")
            
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
