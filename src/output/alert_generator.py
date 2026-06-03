# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from typing import Dict, Any

def generate_driver_whatsapp_alert(shipment_id: str, validated_advisory: Dict[str, Any]) -> str:
    """
    Formally formats the LLM driver alerts into a clear, WhatsApp-friendly text alert
    complete with warning emojis and clear bullets.
    """
    overall_risk = validated_advisory.get("overall_risk", "low").upper()
    alerts = validated_advisory.get("driver_alerts", [])
    
    # Emoji based on risk rating
    if overall_risk == "HIGH":
        emoji = "⚠️ 🛑"
    elif overall_risk == "MEDIUM":
        emoji = "⚠️ 🚛"
    else:
        emoji = "✅ 🚛"
        
    alert_count = len(alerts)
    header = f"{emoji} {overall_risk} RISK — {alert_count} alert flag{'s' if alert_count != 1 else ''} on shipment {shipment_id}:\n"
    
    body = ""
    for idx, alert in enumerate(alerts):
        body += f"{idx + 1}. {alert}\n"
        
    if not alerts:
        body = "1. No immediate compliance or route risk flags detected. Safe travels!\n"
        
    footer = "\nPlease drive safely and adhere to highway compliance instructions."
    return header + body + footer
