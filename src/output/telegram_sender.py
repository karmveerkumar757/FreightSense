# -*- coding: utf-8 -*-
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def send_telegram_alert(chat_id: str, message_body: str) -> dict:
    """
    Sends a message to a Telegram Chat ID using the bot token from environment variables.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token or "your_telegram_bot_token" in bot_token or not bot_token.strip():
        print("⚠️ Telegram Bot Token not configured in .env. Falling back to mock dispatch logging.")
        print(f"\n--- [MOCK TELEGRAM DISPATCH] ---")
        print(f"Chat ID: {chat_id}")
        print(f"Message:\n{message_body}")
        print(f"--------------------------------\n")
        return {"status": "mocked", "message": "Telegram Bot Token not configured. Alert printed to console logs."}
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message_body,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        res_data = response.json()
        if response.status_code == 200 and res_data.get("ok"):
            print(f"✅ Telegram alert sent successfully to Chat ID: {chat_id}")
            return {"status": "success", "sid": f"tg-{res_data['result']['message_id']}"}
        else:
            error_desc = res_data.get("description", "Unknown error")
            print(f"❌ Telegram API send failed: {error_desc}")
            return {"status": "error", "error": error_desc}
    except Exception as e:
        print(f"❌ Telegram HTTP request failed: {e}")
        return {"status": "error", "error": str(e)}
