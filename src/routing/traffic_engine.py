# -*- coding: utf-8 -*-
from datetime import datetime
import pytz

def get_traffic_multiplier(city_name: str, route_time: datetime = None) -> float:
    """
    Simulates a predictive traffic engine based on heuristics.
    Returns a time multiplier (e.g., 1.5x slower) based on city and time-of-day.
    """
    if route_time is None:
        # Default to current IST time
        ist = pytz.timezone('Asia/Kolkata')
        route_time = datetime.now(ist)
        
    hour = route_time.hour
    
    # Base multiplier
    multiplier = 1.0
    
    # Morning Rush Hour (8 AM - 11 AM)
    if 8 <= hour < 11:
        multiplier += 0.4
    # Evening Rush Hour (5 PM - 9 PM)
    elif 17 <= hour < 21:
        multiplier += 0.5
    # Late Night (11 PM - 5 AM) - Fast traffic
    elif hour >= 23 or hour < 5:
        multiplier -= 0.2
        
    # City specific penalties
    city_lower = city_name.lower()
    if city_lower in ["delhi", "mumbai", "bangalore", "bengaluru"]:
        # Top congested cities add an extra 10-20% base delay
        multiplier += 0.15
        if 17 <= hour < 21:
            multiplier += 0.2 # Extreme evening rush hour
            
    elif city_lower in ["gurugram", "gurgaon", "pune", "chennai"]:
        multiplier += 0.1
        
    return max(0.8, multiplier) # Never impossibly fast

def simulate_predictive_traffic(duration_mins: float, cities: list) -> tuple:
    """
    Applies the predictive traffic multiplier to the base OSRM route duration.
    Returns (adjusted_duration_mins, alerts_list).
    """
    if not cities:
        return duration_mins, []
        
    # Take the worst multiplier among the cities on the route
    worst_multiplier = 1.0
    for city in cities:
        mult = get_traffic_multiplier(city)
        if mult > worst_multiplier:
            worst_multiplier = mult
            
    adjusted_duration = duration_mins * worst_multiplier
    alerts = []
    
    if worst_multiplier >= 1.5:
        alerts.append("SEVERE TRAFFIC: Heavy rush hour congestion expected on route.")
    elif worst_multiplier >= 1.2:
        alerts.append("MODERATE TRAFFIC: Some delays expected due to active traffic hours.")
        
    return adjusted_duration, alerts
