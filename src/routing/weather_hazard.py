# -*- coding: utf-8 -*-
import requests

def get_weather_hazards(lat, lon):
    """
    Fetches the current weather for the given coordinates using the free Open-Meteo API.
    Returns a hazard flag string (e.g., 'Heavy Rain', 'Extreme Heat') or None if clear.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            weathercode = current.get("weathercode", 0)
            
            # WMO Weather interpretation codes
            if weathercode in [61, 63, 65, 80, 81, 82]:
                return "Rain/Showers"
            elif weathercode in [71, 73, 75, 85, 86]:
                return "Snow/Blizzard"
            elif weathercode in [95, 96, 99]:
                return "Thunderstorm"
            elif weathercode in [45, 48]:
                return "Dense Fog"
            else:
                return "Clear/Normal"
    except Exception as e:
        print(f"⚠️ Weather API failed for {lat}, {lon}: {e}")
        
    return "Unknown"
