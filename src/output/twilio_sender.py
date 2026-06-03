# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

# Global import flag to catch missing package imports cleanly
_twilio_installed = False
try:
    from twilio.rest import Client
    _twilio_installed = True
except ImportError:
    pass

def send_whatsapp_alert(to_number: str, message_body: str) -> dict:
    """
    Sends a WhatsApp message using Twilio API.
    to_number should be in format '+919999999999' or 'whatsapp:+919999999999'.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER") # e.g. 'whatsapp:+14155238886' (Twilio Sandbox)
    
    if not _twilio_installed or not account_sid or not auth_token or not from_number:
        print("⚠️ Twilio not configured or library missing. Falling back to mock dispatch logging.")
        print(f"\n--- [MOCK WHATSAPP DISPATCH] ---")
        print(f"To: {to_number}")
        print(f"From: {from_number or 'MOCK_SENDER'}")
        print(f"Message:\n{message_body}")
        print(f"---------------------------------\n")
        return {"status": "mocked", "message": "Twilio not configured. Alert printed to console logs."}
        
    # Ensure phone number formats
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"
    if not from_number.startswith("whatsapp:"):
        from_number = f"whatsapp:{from_number}"
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        print(f"✅ Twilio WhatsApp message sent successfully: SID {message.sid}")
        return {"status": "success", "sid": message.sid}
    except Exception as e:
        print(f"❌ Twilio WhatsApp send failed: {e}")
        return {"status": "error", "error": str(e)}

def send_sms_alert(to_number: str, message_body: str) -> dict:
    """
    Sends an SMS message using Twilio API.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER") # e.g. '+1234567890' (Twilio SMS number)
    
    if not _twilio_installed or not account_sid or not auth_token or not from_number:
        print("⚠️ Twilio not configured or library missing. Falling back to mock dispatch logging.")
        print(f"\n--- [MOCK SMS DISPATCH] ---")
        print(f"To: {to_number}")
        print(f"From: {from_number or 'MOCK_SENDER'}")
        print(f"Message:\n{message_body}")
        print(f"---------------------------\n")
        return {"status": "mocked", "message": "Twilio not configured. Alert printed to console logs."}
        
    # Standard numbers for SMS must not have 'whatsapp:' prefix
    if to_number.startswith("whatsapp:"):
        to_number = to_number.replace("whatsapp:", "")
    if from_number.startswith("whatsapp:"):
        from_number = from_number.replace("whatsapp:", "")
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        print(f"✅ Twilio SMS message sent successfully: SID {message.sid}")
        return {"status": "success", "sid": message.sid}
    except Exception as e:
        print(f"❌ Twilio SMS send failed: {e}")
        return {"status": "error", "error": str(e)}
