# -*- coding: utf-8 -*-
import sys
import os

# Force Hugging Face offline mode to avoid slow/hanging online checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Set page config
st.set_page_config(
    page_title="FreightSense — AI Logistics Advisory",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_URL = "http://localhost:8000"

# Predefined City Coordinates for interactive maps
CITY_COORDINATES = {
    "delhi": [28.6139, 77.2090],
    "gurugram": [28.4595, 77.0266],
    "gurgaon": [28.4595, 77.0266],
    "mumbai": [19.0760, 72.8777],
    "pune": [18.5204, 73.8567],
    "bangalore": [12.9716, 77.5946],
    "bengaluru": [12.9716, 77.5946],
    "chennai": [13.0827, 80.2707],
    "ahmedabad": [23.0225, 72.5714],
    "jaipur": [26.9124, 75.7873],
    "kolkata": [22.5726, 88.3639],
    "hyderabad": [17.3850, 78.4867]
}

def get_road_route_details(points):
    """
    Given a list of [lat, lon] coordinates, query OSRM to get:
    - road coordinates: list of [lat, lon]
    - distance: float (in kilometers)
    - duration: float (in minutes)
    Falls back to straight lines and 0.0 stats if the API request fails.
    """
    if len(points) < 2:
        return points, 0.0, 0.0
        
    coord_strings = [f"{lon},{lat}" for lat, lon in points]
    coords_path = ";".join(coord_strings)
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_path}?geometries=geojson&overview=full"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                geometry = route["geometry"]
                coordinates = geometry["coordinates"]
                distance_km = route.get("distance", 0.0) / 1000.0
                duration_mins = route.get("duration", 0.0) / 60.0
                
                # OSRM returns [lon, lat], convert to [lat, lon] for Folium
                folium_coords = [[lat, lon] for lon, lat in coordinates]
                return folium_coords, distance_km, duration_mins
    except Exception as e:
        print(f"⚠️ OSRM routing failed: {e}. Falling back to straight-line path.")
        
    return points, 0.0, 0.0

# Inject Custom CSS for Premium Dark Glassmorphic Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Stylings */
    .title-text {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem !important;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #8D99AE;
        font-weight: 300;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }
    
    /* Alert cards depending on status */
    .risk-high {
        border-left: 5px solid #FF6B6B !important;
        background: rgba(255, 107, 107, 0.05);
    }
    .risk-medium {
        border-left: 5px solid #FFD93D !important;
        background: rgba(255, 217, 61, 0.05);
    }
    .risk-low {
        border-left: 5px solid #6BCB77 !important;
        background: rgba(107, 203, 119, 0.05);
    }
    
    /* Custom chip stylings */
    .entity-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }
    .label-cargo { background-color: #4D96FF; }
    .label-time { background-color: #FF6B6B; }
    .label-route { background-color: #6BCB77; }
    .label-handling { background-color: #9B5DE5; }
    .label-compliance { background-color: #F15BB5; }
    
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='text-align: center;'>🚛 FreightSense Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Main Navigation Tab selector
nav = st.sidebar.radio("Navigate", ["Advisory Console", "Active Shipments Log", "Compliance Knowledge Hub"])

# Display Title
st.markdown("<h1 class='title-text'>FreightSense</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>AI-Powered Logistics Constraint & Compliance Risk Advisory Engine</p>", unsafe_allow_html=True)

# ----------------- Navigation 1: Advisory Console -----------------
if nav == "Advisory Console":
    st.markdown("### 📥 Document & Note Ingestion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        input_type = st.radio("Select Input Source:", ["Manual Text Dispatcher Entry", "Upload Freight PDF / Challan", "Upload Audio Voice Note"])
        
        raw_text_input = ""
        uploaded_file = None
        
        if input_type == "Manual Text Dispatcher Entry":
            raw_text_input = st.text_area(
                "Enter unstructured freight instructions (Hindi / English mixed):",
                placeholder="Example: Urgent medicines delivery AIIMS Delhi bhejna hai before 8 AM via NH-48. Keep frozen. e-way bill checked.",
                height=150
            )
        elif input_type == "Upload Freight PDF / Challan":
            uploaded_file = st.file_uploader("Choose a PDF document or scan image", type=["pdf", "png", "jpg", "jpeg"])
        elif input_type == "Upload Audio Voice Note":
            uploaded_file = st.file_uploader("Choose a WhatsApp voice note or audio file", type=["wav", "mp3", "m4a", "ogg", "aac"])
            
        process_btn = st.button("🚀 Analyze Shipment & Generate Advisory", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("💡 **System capabilities:**")
        st.write("1. **Digital & Scanned Ingestion:** PyMuPDF + Tesseract OCR handles scanned paper slips.")
        st.write("2. **Whisper Transcription:** Transcribes voice instructions in mixed Hinglish.")
        st.write("3. **Constraint Extraction:** NER models extract cargo, times, routes, handling rules.")
        st.write("4. **ChromaDB RAG:** Validates details against Motor Vehicles Act and local city timings.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Trigger Pipeline API
    if process_btn:
        result = None
        with st.spinner("Processing multi-modal inputs, running NER extraction & querying compliance RAG..."):
            try:
                if uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/extract", files=files)
                elif raw_text_input:
                    data = {"text": raw_text_input}
                    response = requests.post(f"{API_URL}/extract", data=data)
                else:
                    st.warning("Please provide input text or upload a file.")
                    response = None
                    
                if response and response.status_code == 200:
                    result = response.json()
                    st.session_state["active_result"] = result
                    st.success("Advisory generated successfully!")
                elif response:
                    st.error(f"Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}. Is the FastAPI server running?")
                
    # Display Active Results
    if "active_result" in st.session_state:
        res = st.session_state["active_result"]
        shipment_id = res["shipment_id"]
        raw_text = res["raw_text"]
        entities = res["entities"]
        intents = res["intents"]
        advisory = res["advisory"]
        whatsapp_alert = res["driver_whatsapp_alert"]
        
        st.markdown("---")
        st.markdown(f"## 📋 Compliance Evaluation: {shipment_id}")
        
        # Advisory Summary
        overall_risk = advisory.get("overall_risk", "low").lower()
        risk_class = f"glass-card risk-{overall_risk}"
        
        st.markdown(f"""
        <div class='{risk_class}'>
            <h3>⚠️ Risk Severity: {overall_risk.upper()}</h3>
            <p style='font-size: 1.1rem; font-style: italic;'>"{advisory.get('plain_english_advisory')}"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Details Columns
        det_col1, det_col2 = st.columns([1, 1])
        
        with det_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🪵 Normalized Ingestion Text")
            st.info(raw_text)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Intent Classification & Entities")
            
            st.markdown("**Sentence Intents:**")
            for it in intents:
                st.write(f"- *\"{it['sentence']}\"* → `{it['intent']}`")
                
            st.markdown("<br>**Extracted Constraint Tokens:**", unsafe_allow_html=True)
            if not entities:
                st.write("No named entities detected.")
            for ent in entities:
                label = ent["label"].lower()
                chip_class = "entity-chip"
                if "cargo" in label:
                    chip_class += " label-cargo"
                elif "time" in label:
                    chip_class += " label-time"
                elif "route" in label:
                    chip_class += " label-route"
                elif "handling" in label:
                    chip_class += " label-handling"
                else:
                    chip_class += " label-compliance"
                    
                st.markdown(f"<span class='{chip_class}'>{ent['label']}: {ent['text']}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with det_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### 🛡️ Compliance & Route Violations")
            
            risks = advisory.get("risks", [])
            if not risks:
                st.success("No compliance or timing risks flagged.")
            for r in risks:
                sev = r.get("severity", "low").upper()
                st.markdown(f"- **[{sev}] {r.get('type').upper()}:** {r.get('description')}")
                
            st.markdown("<br>**Mandatory Compliance Permits:**", unsafe_allow_html=True)
            permits = advisory.get("compliance", {}).get("permits_required", [])
            if not permits:
                st.write("No transit permits required.")
            for p in permits:
                st.write(f"-  {p}")
                
            st.write(f"- **Dangerous Goods Flag:** `{'YES' if advisory.get('compliance', {}).get('dangerous_goods') else 'NO'}`")
            st.write(f"- **e-Way Bill Expiry:** `{advisory.get('compliance', {}).get('eway_bill_valid_until') or 'None'}`")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # WhatsApp bot format
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📱 Driver WhatsApp Alert (Twilio Payload Preview)")
        st.code(whatsapp_alert, language="text")
        
        # SMS / WhatsApp / Telegram Dispatch Controls
        st.markdown("<br><h5>📲 Send Real-Time Alert to Driver</h5>", unsafe_allow_html=True)
        col_phone, col_channel = st.columns([2, 1])
        with col_channel:
            alert_channel = st.selectbox("Dispatch Channel:", ["WhatsApp", "SMS", "Telegram"])
            
        with col_phone:
            if alert_channel == "Telegram":
                input_label = "Driver Telegram Chat ID:"
                input_placeholder = "e.g. 987654321"
            else:
                input_label = "Driver Phone Number (with country code):"
                input_placeholder = "e.g. +919876543210"
            driver_phone = st.text_input(input_label, placeholder=input_placeholder, key="driver_phone_input")
            
        send_alert_btn = st.button("🚀 Dispatch Alert", use_container_width=True)
        if send_alert_btn:
            if not driver_phone.strip():
                if alert_channel == "Telegram":
                    st.warning("Please enter a valid Chat ID.")
                else:
                    st.warning("Please enter a valid phone number.")
            else:
                try:
                    payload = {
                        "phone_number": driver_phone.strip(),
                        "alert_type": alert_channel.lower()
                    }
                    send_res = requests.post(f"{API_URL}/send_alert/{shipment_id}", json=payload)
                    if send_res.status_code == 200:
                        res_data = send_res.json()
                        if res_data.get("status") == "success":
                            st.success(f"✅ Alert dispatched successfully! Reference: {res_data.get('sid')}")
                        elif res_data.get("status") == "mocked":
                            st.info(f"💡 Mock Mode: {res_data.get('message')}")
                        else:
                            st.error(f"❌ Error sending message: {res_data.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ API returned error status {send_res.status_code}: {send_res.text}")
                except Exception as e:
                    st.error(f"❌ Failed to dispatch alert: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive Map Section
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ Route Constraint Visualization Map")
        
        # Parse cities from text in the order they appear to draw route direction correctly
        text_lower = raw_text.lower()
        city_positions = []
        seen_coords = set()
        for city, coords in CITY_COORDINATES.items():
            pos = text_lower.find(city)
            if pos != -1:
                coord_tuple = tuple(coords)
                if coord_tuple not in seen_coords:
                    seen_coords.add(coord_tuple)
                    city_positions.append((pos, city))
                    
        # Sort by their occurrence position in the text
        city_positions.sort()
        route_cities = [city for pos, city in city_positions]
                
        if not route_cities:
            st.info("ℹ️ No recognized route nodes (e.g. Delhi, Gurugram, Mumbai, Pune, Bangalore, Jaipur) were detected in the text to display map route markers.")
        else:
            center_coords = CITY_COORDINATES[route_cities[0]]
            m = folium.Map(location=center_coords, zoom_start=6, tiles="OpenStreetMap")
            
            # Add city markers
            points = []
            for city in route_cities:
                coords = CITY_COORDINATES[city]
                points.append(coords)
                folium.Marker(
                    location=coords,
                    popup=f"Shipment Node: {city.upper()}",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)
                
            # Draw a route line if we have multiple cities
            if len(points) >= 2:
                # Query OSRM for actual road route and metrics
                road_points, distance_km, duration_mins = get_road_route_details(points)
                
                # Display route statistics in a premium banner
                if distance_km > 0:
                    hours = int(duration_mins // 60)
                    mins = int(duration_mins % 60)
                    time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
                    st.markdown(f"""
                    <div style='background: rgba(77, 150, 255, 0.08); border: 1px solid rgba(77, 150, 255, 0.2); padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem;'>
                        📍 <b>Calculated Road Route:</b> Distance: <b>{distance_km:.1f} km</b> | Est. Travel Time: <b>{time_str}</b>
                    </div>
                    """, unsafe_allow_html=True)
                
                folium.PolyLine(road_points, color="#4D96FF", weight=5, opacity=0.8).add_to(m)
                
            # Highlight Delhi Ring Road Risk if Delhi is present
            if "delhi" in route_cities:
                # Highlight restricted zone around ring road (centered on central Delhi)
                folium.Circle(
                    location=[28.6139, 77.2090],
                    radius=15000,
                    color="#FF6B6B",
                    fill=True,
                    fill_color="#FF6B6B",
                    fill_opacity=0.15,
                    popup="DELHI TRUCK BAN ZONE: Banned 7AM-11PM"
                ).add_to(m)
                
            # Highlight Gurugram Toll risk if Gurgaon/Gurugram is present
            if "gurgaon" in route_cities or "gurugram" in route_cities:
                folium.Circle(
                    location=[28.4595, 77.0266],
                    radius=3000,
                    color="#FFD93D",
                    fill=True,
                    fill_color="#FFD93D",
                    fill_opacity=0.2,
                    popup="NH-48 TOLL PLAZA: High congestion point"
                ).add_to(m)
                
            st_folium(m, width=1200, height=400)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # MLOps feedback panel
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔁 MLOps Feedback Loop (Report Incorrect Advisories)")
        
        with st.form("feedback_form"):
            feedback_notes = st.text_area("Detail any incorrect entity tags, false risk warnings, or general dispatcher overrides:")
            
            # Avoid duplicate option names in selectbox list
            risk_options = ["LOW", "MEDIUM", "HIGH"]
            current_risk_upper = overall_risk.upper()
            default_idx = risk_options.index(current_risk_upper) if current_risk_upper in risk_options else 0
            override_risk = st.selectbox("Corrected Risk Severity (if incorrect):", risk_options, index=default_idx)
            
            submit_feedback_btn = st.form_submit_button("💾 Submit Feedback to MLOps DB")
            
            if submit_feedback_btn:
                try:
                    corrected_data = advisory.copy()
                    corrected_data["overall_risk"] = override_risk.lower()
                    
                    feedback_payload = {
                        "feedback_notes": feedback_notes,
                        "corrected_data": corrected_data
                    }
                    fb_res = requests.post(f"{API_URL}/feedback/{shipment_id}", json=feedback_payload)
                    if fb_res.status_code == 200:
                        st.success("Feedback saved successfully! This record is flagged for MLOps retraining.")
                    else:
                        st.error("Failed to log feedback.")
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Navigation 2: Active Shipments Log -----------------
elif nav == "Active Shipments Log":
    st.markdown("### 🗃️ Shipment Compliance History Log")
    
    try:
        response = requests.get(f"{API_URL}/shipments")
        if response.status_code == 200:
            shipments = response.json()
            
            if not shipments:
                st.info("No shipments logged in the database yet.")
                
            for s in shipments:
                # Severity-specific style
                risk_lvl = s.get("overall_risk", "low").lower()
                card_class = f"glass-card risk-{risk_lvl}"
                
                st.markdown(f"""
                <div class='{card_class}'>
                    <div style='display: flex; justify-content: space-between;'>
                        <h4>ID: {s['id']}</h4>
                        <span style='font-weight: bold;'>Risk: {risk_lvl.upper()}</span>
                    </div>
                    <p style='color: #8D99AE; font-size: 0.85rem;'>Logged at: {s['timestamp']}</p>
                    <p style='font-size: 0.95rem;'><strong>Text:</strong> {s['raw_input_text']}</p>
                    <p style='font-size: 0.95rem; font-style: italic;'><strong>Advisory:</strong> "{s['advisory_json'].get('plain_english_advisory', 'No advisory summary.')}"</p>
                    {"<p style='color: #FFD93D; font-size: 0.9rem;'>⚠️ <strong>Feedback Logged:</strong> " + str(s.get('feedback_notes') or '') + "</p>" if s.get('has_feedback') else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Failed to fetch shipments log.")
    except Exception as e:
        st.error(f"API Connection error: {e}")

# ----------------- Navigation 3: Compliance Knowledge Hub -----------------
elif nav == "Compliance Knowledge Hub":
    st.markdown("### 🔍 Regulations Knowledge Hub (ChromaDB Vector Store)")
    st.markdown("<p style='color: #8D99AE;'>Query the semantic vector store containing Indian transport acts, local city timing codes, and road safety manuals.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    search_query = st.text_input("Enter query to search regulations (e.g. 'eway bill validity', 'overloading fine', 'delhi truck ban'):")
    search_btn = st.button("🔍 Search RAG Database")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if search_btn and search_query:
        # We can call the backend RAG module directly or expose an endpoint.
        # Calling retriever module directly:
        try:
            sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
            from src.genai.rag_retriever import retrieve_context
            
            with st.spinner("Searching semantic database..."):
                results = retrieve_context(search_query, n_results=4)
                
                if not results:
                    st.warning("No matching regulatory context found.")
                else:
                    st.markdown("#### Retrieved Regulatory Contexts:")
                    for idx, chunk in enumerate(results):
                        st.markdown(f"""
                        <div class='glass-card'>
                            <h5>Regulation Excerpt #{idx+1}</h5>
                            <p style='font-size: 0.95rem;'>{chunk}</p>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to query local vector store: {e}")
