# Telegram Bot Alert Dispatch Integration Plan

This plan outlines the steps to create a Telegram Bot and integrate it into the FreightSense system as a free, real-time alert dispatch channel.

---

## Part 1: How to Setup Your Telegram Bot (No Code Required)

To use Telegram for alerts, you first need to create a bot on Telegram and get your target Chat ID. Follow these steps:

### Step 1: Create the Telegram Bot
1. Open the **Telegram** application on your phone or computer.
2. Search for the user `@BotFather` (the official verified bot-creation bot).
3. Click **Start** and send the message: `/newbot`
4. Choose a display name for your bot (e.g., `FreightSense Alerts`).
5. Choose a username for your bot. It must end in `bot` (e.g., `freightsense_alert_bot`).
6. BotFather will send you a message containing your **HTTP API Access Token**.
   * *It looks like:* `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`
   * Keep this token secure. This is your `TELEGRAM_BOT_TOKEN`.

### Step 2: Activate the Bot
1. Click the link to your new bot provided by BotFather (e.g., `t.me/freightsense_alert_bot`).
2. Click **Start** to open a conversation with it. (The bot cannot send you messages until you start a chat with it).

### Step 3: Get Your Chat ID
1. Search for `@userinfobot` or `@GetIDBot` in Telegram.
2. Send any message to it. It will reply with your **Id** (a sequence of numbers like `987654321`). This is the ID for your Telegram account to receive push alerts.

---

## Part 2: Proposed Code Changes

We will integrate Telegram alerts by modifying the configuration, backend API routing, and frontend UI controls.

### 1. Configuration

#### [MODIFY] [.env](file:///d:/Coding/FreightSense_Project/.env)
Add the Telegram bot token variable:
```ini
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

---

### 2. Output Layer

#### [NEW] [telegram_sender.py](file:///d:/Coding/FreightSense_Project/src/output/telegram_sender.py)
Create a new utility file containing the logic for sending messages using the standard Telegram Bot HTTP API:
```python
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
```

---

### 3. Backend Integration

#### [MODIFY] [main.py](file:///d:/Coding/FreightSense_Project/main.py)
Update the `/send_alert/{shipment_id}` API endpoint to support the `"telegram"` dispatch type:
```python
class SendAlertRequest(BaseModel):
    phone_number: str # Will represent phone number for SMS/WhatsApp, or Chat ID for Telegram
    alert_type: str = "whatsapp"  # whatsapp, sms, or telegram

# ... in send_shipment_alert:
        if request.alert_type.lower() == "whatsapp":
            res = send_whatsapp_alert(request.phone_number, alert_body)
        elif request.alert_type.lower() == "sms":
            res = send_sms_alert(request.phone_number, alert_body)
        elif request.alert_type.lower() == "telegram":
            from src.output.telegram_sender import send_telegram_alert
            res = send_telegram_alert(request.phone_number, alert_body)
```

---

### 4. Frontend Integration

#### [MODIFY] [app.py](file:///d:/Coding/FreightSense_Project/app.py)
Update the dropdown options to allow selecting "Telegram" and dynamically change the text input label from "Driver Phone Number" to "Driver Telegram Chat ID":
* Add `"Telegram"` to `alert_channel` selectbox.
* Change label of text input dynamically:
  ```python
  if alert_channel == "Telegram":
      input_label = "Driver Telegram Chat ID:"
      input_placeholder = "e.g. 987654321"
  else:
      input_label = "Driver Phone Number (with country code):"
      input_placeholder = "e.g. +919876543210"
  ```

---

## Verification Plan

### Manual Verification
1. Open the updated Streamlit Dashboard.
2. Select **Telegram** as the dispatch channel.
3. Verify that the label and placeholder change dynamically to ask for a Chat ID.
4. Try to click send *without* a bot token in `.env`. Verify it outputs the Mock dispatch correctly to the terminal console logs.
5. Add a real bot token to `.env` and enter your personal Chat ID. Verify that your phone/desktop Telegram client receives the alert instantly in markdown format.
