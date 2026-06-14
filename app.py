# -*- coding: utf-8 -*-
import sys
import os

# Force Hugging Face offline mode to avoid slow/hanging online checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import requests
import pandas as pd
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
nav = st.sidebar.radio("Navigate", ["Advisory Console", "Active Shipments Log", "Compliance Knowledge Hub", "Analytics & Insights"])

# -- Sidebar: Settings --
st.sidebar.header("⚙️ Configuration")
# We remove manual Gemini key entry since it's loaded from env but keep it for flexibility
gemini_key = st.sidebar.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")

from src.ingestion.bhashini_asr import SUPPORTED_LANGUAGES
driver_language = st.sidebar.selectbox(
    "Driver's Language (Voice Readback)",
    options=list(SUPPORTED_LANGUAGES.keys()),
    format_func=lambda x: SUPPORTED_LANGUAGES[x],
    index=0  # Default Hindi
)

# -- Main UI --
st.title("🚚 FreightSense")
st.markdown("### AI-Powered Logistics Constraint & Compliance Risk Advisory Engine (v5.0)")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Instruction Entry", "📊 Analysis Dashboard", "📈 DPO Feedback", "🗺️ Route Map"])

with tab1:
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
                    st.success(f"✅ Processing complete in {result.get('processing_time_ms', 0):.0f} ms!")
                    
                    # Switch to dashboard tab to show results (Streamlit 1.32 supports this via state, but we'll just show it here)
                    st.session_state.extraction_result = result
                    
                    with st.expander("Raw JSON Response"):
                        st.json(result)
                elif response:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}. Is the FastAPI server running?")
                
if "extraction_result" in st.session_state:
    with tab2:
        res = st.session_state.extraction_result
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Constraints Extracted")
            st.json(res.get("constraints", {}))
            
        with col2:
            st.subheader("LSTM Delay Prediction")
            delay = res.get("delay_prediction", {})
            st.metric("Delay Probability", f"{delay.get('delay_probability', 0)*100:.1f}%")
            st.info(delay.get('recommendation', ''))
            
        st.divider()
        
        # SHAP Explainability Chart
        st.subheader("XGBoost Risk Assessment")
        try:
            from frontend.components.shap_chart import render_shap_chart
            render_shap_chart(res.get("risk_explanation", {}))
        except Exception as e:
            st.write("SHAP Component loading... (ensure requirements are met)")
            st.json(res.get("risk_explanation", {}))
            
        st.divider()
        
        st.subheader("🤖 Advisory")
        advisory = res.get("advisory", {})
        st.markdown(f"**Overall Risk:** {advisory.get('overall_risk', 'UNKNOWN')}")
        st.markdown(f"**Advisory:** {advisory.get('advisory_text', '')}")
        
        if advisory.get("reasoning_chain"):
            with st.expander("Agentic RAG Reasoning Chain"):
                st.json(advisory["reasoning_chain"])
                
        # Audio Readback
        if st.button("🔊 Generate Audio Advisory (Bhashini)"):
            try:
                from src.output.bhashini_tts import BhashiniTTS
                tts = BhashiniTTS()
                with st.spinner(f"Translating and generating audio in {SUPPORTED_LANGUAGES[driver_language]}..."):
                    audio_bytes = tts.translate_and_speak(advisory.get('advisory_text', ''), driver_language)
                    
                if audio_bytes:
                    st.session_state.generated_audio = audio_bytes
                else:
                    st.error("Audio generation failed.")
            except Exception as e:
                st.error(f"Bhashini error: {e}")
                
        # If audio is generated, show the player and the "Send to Driver" section
        if "generated_audio" in st.session_state:
            st.audio(st.session_state.generated_audio, format="audio/wav")
            st.success(f"Generated audio in {SUPPORTED_LANGUAGES[driver_language]}")
            
            st.markdown("---")
            st.subheader("📲 Send to Driver (Manual via WhatsApp Web)")
            st.markdown("<p style='font-size: 0.85rem; color: #8D99AE;'>Since there is no paid WhatsApp API configured, use this to send manually from your own phone/WhatsApp Web.</p>", unsafe_allow_html=True)
            
            col_ph1, col_ph2 = st.columns([1, 1])
            with col_ph1:
                driver_phone = st.text_input("Driver's WhatsApp Number:", value="+91")
                
                # Clean phone number for WhatsApp link
                clean_phone = ''.join(filter(str.isdigit, driver_phone))
                if driver_phone.startswith('+') and not clean_phone.startswith('+'):
                    clean_phone = '+' + clean_phone
                    
                import urllib.parse
                whatsapp_text = urllib.parse.quote(f"🚛 *FreightSense Advisory*\n\n{advisory.get('advisory_text', '')}\n\n_Please listen to the attached audio instruction._")
                wa_link = f"https://wa.me/{clean_phone}?text={whatsapp_text}"
                
                if len(clean_phone) > 10:
                    st.markdown(f'<a href="{wa_link}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background-color: #25D366; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; width: 100%; text-align: center;">💬 Open WhatsApp</a>', unsafe_allow_html=True)
                else:
                    st.button("💬 Open WhatsApp (Enter valid number)", disabled=True, use_container_width=True)
                    
            with col_ph2:
                st.markdown("<br>", unsafe_allow_html=True) # padding to align with text input
                st.download_button(
                    label="📥 Download Audio Voice Note",
                    data=st.session_state.generated_audio,
                    file_name="freightsense_advisory.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
                
            st.info("💡 **Jugaad Guide:** 1. Download the Audio. 2. Click 'Open WhatsApp' to send the text message. 3. Drag and drop the downloaded audio into the WhatsApp chat!")

    with tab3:
        st.subheader("DPO Alignment Feedback")
        st.markdown("Help train our local Gemma-2B model by editing the advisory if it wasn't perfect.")
        
        res = st.session_state.extraction_result
        current_advisory = res.get("advisory", {}).get("advisory_text", "")
        
        chosen_advisory = st.text_area("Edit Advisory to your preference:", value=current_advisory, height=150)
        rejection_reason = st.text_input("Reason for edit (optional):")
        
        if st.button("Submit DPO Feedback Pair"):
            try:
                import sqlite3
                conn = sqlite3.connect("data/freightsense.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM preference_pairs")
                count = cursor.fetchone()[0]
                conn.close()
                
                # Send to API
                payload = {
                    "shipment_id": res.get("shipment_id", "unknown"),
                    "constraints": res.get("constraints", {}),
                    "rejected_advisory": current_advisory,
                    "chosen_advisory": chosen_advisory,
                    "rejection_reason": rejection_reason
                }
                
                import requests
                requests.post("http://localhost:8000/feedback", json=payload)
                
                st.success("Feedback submitted! Background LLM-as-Judge evaluation queued.")
                
                progress = min(count / 100.0, 1.0)
                st.progress(progress)
                st.success(f"Feedback logged successfully! Total pairs collected: {count + 1}")
            except Exception as e:
                st.error(f"Failed to submit feedback: {e}")

    with tab4:
        st.subheader("🗺️ Multi-Objective Route Optimization")
        st.markdown("Visualizing the optimal route balancing distance, compliance risk, and predicted delays.")
        
        res = st.session_state.extraction_result
        locations_text = res.get("constraints", {}).get("locations", [])
        
        if not locations_text or len(locations_text) < 2:
            st.info("⚠️ Not enough locations extracted to generate a route. Please ensure the instruction mentions at least a source and destination (e.g., 'Delhi to Pune').")
        else:
            st.write(f"**Extracted Locations:** {', '.join(locations_text)}")
            
            # Map strings to coordinates
            mapped_locations = []
            for loc in locations_text:
                loc_lower = loc.lower()
                if loc_lower in CITY_COORDINATES:
                    coords = CITY_COORDINATES[loc_lower]
                    mapped_locations.append({"name": loc, "lat": coords[0], "lon": coords[1]})
                    
            if len(mapped_locations) < 2:
                st.warning("Could not map extracted locations to coordinates. Currently supporting major cities like Delhi, Mumbai, Pune, etc.")
            else:
                if st.button("Generate Optimized Route", key="gen_route"):
                    with st.spinner("Calculating optimal multi-objective path via VRP Engine..."):
                        try:
                            # 1. Call Backend VRP Endpoint
                            payload = {
                                "instruction": "VRP Route Generation", 
                                "locations": mapped_locations
                            }
                            vrp_response = requests.post(f"{API_URL}/optimise_route", json=payload)
                            
                            if vrp_response.status_code == 200:
                                vrp_data = vrp_response.json()
                                best_route = vrp_data.get("best_route", {})
                                route_indices = best_route.get("route", [])
                                
                                # Reorder locations based on VRP result
                                if route_indices:
                                    ordered_locs = [mapped_locations[i] for i in route_indices if i < len(mapped_locations)]
                                else:
                                    ordered_locs = mapped_locations
                                    
                                # 2. Get Road Polyline from OSRM
                                points = [[loc["lat"], loc["lon"]] for loc in ordered_locs]
                                route_coords, dist_km, dur_mins = get_road_route_details(points)
                                
                                # Save to session state to prevent map from disappearing on interact
                                st.session_state.route_data = {
                                    "ordered_locs": ordered_locs,
                                    "route_coords": route_coords,
                                    "dist_km": dist_km,
                                    "dur_mins": dur_mins,
                                    "selection_reason": best_route.get("selection_reason", "VRP").split('(')[0]
                                }
                                
                            else:
                                st.error(f"VRP Engine failed: {vrp_response.text}")
                        except Exception as e:
                            st.error(f"Failed to connect to VRP API: {e}")
                            
                # Render Map if data exists in session state
                if "route_data" in st.session_state:
                    rd = st.session_state.route_data
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Distance", f"{rd['dist_km']:.1f} km")
                    col2.metric("Est. Duration", f"{rd['dur_mins']:.0f} mins")
                    col3.metric("Selected By", rd['selection_reason'])
                    
                    # 3. Draw Folium Map
                    points = [[loc["lat"], loc["lon"]] for loc in rd['ordered_locs']]
                    m = folium.Map(location=[points[0][0], points[0][1]], zoom_start=6)
                    
                    # Add polyline
                    if len(rd['route_coords']) > 1:
                        folium.PolyLine(locations=rd['route_coords'], color="#4D96FF", weight=5, opacity=0.8).add_to(m)
                        
                    # Add markers
                    for idx, loc in enumerate(rd['ordered_locs']):
                        icon_color = "green" if idx == 0 else ("red" if idx == len(rd['ordered_locs'])-1 else "blue")
                        folium.Marker(
                            location=[loc["lat"], loc["lon"]],
                            popup=f"{idx+1}. {loc['name']}",
                            icon=folium.Icon(color=icon_color, icon="info-sign")
                        ).add_to(m)
                        
                    # Render map in Streamlit
                    st_folium(m, width=800, height=500)
else:
    with tab2:
        st.info("🚀 Welcome! The dashboard is currently empty. Please go back to the '📝 Instruction Entry' tab, enter a freight instruction, and click 'Analyze Shipment' to view the results here.")
        
    with tab3:
        st.info("🚀 Please analyze a shipment first before providing feedback.")
        
    with tab4:
        st.info("🚀 Please analyze a shipment first to generate a route map.")

# ----------------- Navigation 2: Active Shipments Log -----------------
if nav == "Active Shipments Log":
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
if nav == "Compliance Knowledge Hub":
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

# ----------------- Navigation 4: Analytics & Insights -----------------
if nav == "Analytics & Insights":
    st.markdown("### 📊 Executive Analytics Dashboard")
    st.markdown("<p style='color: #8D99AE;'>Overview of shipment compliance, risk distribution, and pipeline drift.</p>", unsafe_allow_html=True)
    
    try:
        response = requests.get(f"{API_URL}/shipments")
        if response.status_code == 200:
            shipments = response.json()
            if not shipments:
                st.info("Not enough data to display analytics. Process more shipments first.")
            else:
                import altair as alt
                # Convert to pandas DataFrame
                df = pd.DataFrame(shipments)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                
                # KPIs
                total_shipments = len(df)
                high_risk = len(df[df['overall_risk'] == 'high'])
                feedback_count = len(df[df['has_feedback'] == True])
                drift_rate = (feedback_count / total_shipments) * 100 if total_shipments > 0 else 0
                
                # NLP & Delay Analysis
                # Parse constraints to get top cargo and delay probs
                cargo_list = []
                avg_delay_prob = 0.0
                delay_count = 0
                for _, row in df.iterrows():
                    # Parse Delay
                    if isinstance(row.get('advisory_json'), dict):
                        # Attempt to infer delay (we don't store raw delay directly in DB schema right now, 
                        # but we can check if there's any mention of delay in constraints or just mock it based on risk)
                        pass
                        
                    # Parse Cargo
                    c = row.get('extracted_constraints')
                    if isinstance(c, dict) and 'cargo_type' in c:
                        for cargo in c['cargo_type']:
                            if cargo: cargo_list.append(cargo.capitalize())
                            
                # Mock average fleet delay based on high risk ratio for demonstration
                avg_delay_prob = (high_risk / total_shipments) * 100 if total_shipments else 0.0
                
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                kpi1.metric("Total Shipments", total_shipments)
                kpi2.metric("High Risk Violations", high_risk, delta_color="inverse")
                kpi3.metric("MLOps Feedbacks", feedback_count)
                kpi4.metric("Pipeline Drift", f"{drift_rate:.1f}%", f"Target < 5%", delta_color="inverse")
                kpi5.metric("Avg Fleet Delay Risk", f"{avg_delay_prob:.1f}%")
                
                st.markdown("---")
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("#### 🔴 Risk Severity Distribution")
                    st.markdown("<p style='font-size: 0.85rem; color: #8D99AE;'>Donut chart of compliance risk levels across all shipments.</p>", unsafe_allow_html=True)
                    
                    risk_counts = df['overall_risk'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Level', 'Count']
                    risk_counts['Risk Level'] = risk_counts['Risk Level'].str.upper()
                    
                    # Altair Donut Chart
                    color_scale = alt.Scale(domain=['HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'], 
                                            range=['#FF6B6B', '#FFD93D', '#6BCB77', '#8D99AE'])
                                            
                    donut = alt.Chart(risk_counts).mark_arc(innerRadius=60, cornerRadius=5).encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Risk Level", type="nominal", scale=color_scale),
                        tooltip=["Risk Level", "Count"]
                    ).properties(height=300).interactive()
                    
                    st.altair_chart(donut, use_container_width=True)
                    
                with col_chart2:
                    st.markdown("#### 📅 Shipment Processing Volume")
                    st.markdown("<p style='font-size: 0.85rem; color: #8D99AE;'>Area chart of shipments processed per day.</p>", unsafe_allow_html=True)
                    volume_counts = df.groupby('date').size().reset_index(name='Volume')
                    volume_counts['date'] = pd.to_datetime(volume_counts['date'])
                    
                    # Altair Area Chart
                    area_chart = alt.Chart(volume_counts).mark_area(
                        line={'color':'#4D96FF'},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='#4D96FF', offset=0),
                                   alt.GradientStop(color='rgba(77, 150, 255, 0)', offset=1)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('Volume:Q', title='Shipments Processed'),
                        tooltip=['date', 'Volume']
                    ).properties(height=300).interactive()
                    
                    st.altair_chart(area_chart, use_container_width=True)
                    
                st.markdown("---")
                col_bottom1, col_bottom2 = st.columns(2)
                
                with col_bottom1:
                    st.markdown("#### 📦 Top Cargo Types Extracted (NER)")
                    st.markdown("<p style='font-size: 0.85rem; color: #8D99AE;'>Most frequently shipped materials identified by the AI.</p>", unsafe_allow_html=True)
                    
                    if cargo_list:
                        cargo_df = pd.DataFrame(cargo_list, columns=['Cargo']).value_counts().reset_index(name='Count').head(5)
                        cargo_bar = alt.Chart(cargo_df).mark_bar(cornerRadiusEnd=4, color="#FF9F29").encode(
                            x=alt.X('Count:Q'),
                            y=alt.Y('Cargo:N', sort='-x'),
                            tooltip=['Cargo', 'Count']
                        ).properties(height=250)
                        st.altair_chart(cargo_bar, use_container_width=True)
                    else:
                        st.info("No distinct cargo types extracted yet.")
                        
                with col_bottom2:
                    st.markdown("#### 🏙️ Top Congested Logistics Hubs")
                    st.markdown("<p style='font-size: 0.85rem; color: #8D99AE;'>Frequency of destination hubs mentioned in instructions.</p>", unsafe_allow_html=True)
                    
                    # Extract cities from text
                    city_counts = {city: 0 for city in CITY_COORDINATES.keys()}
                    for text in df['raw_input_text'].dropna():
                        t_lower = text.lower()
                        for city in city_counts.keys():
                            if city in t_lower:
                                city_counts[city] += 1
                                
                    # Filter zero counts and plot
                    active_cities = {k.capitalize(): v for k, v in city_counts.items() if v > 0}
                    if active_cities:
                        cities_df = pd.DataFrame(list(active_cities.items()), columns=['City', 'Mentions'])
                        cities_bar = alt.Chart(cities_df).mark_bar(cornerRadiusEnd=4, color="#6BCB77").encode(
                            x=alt.X('Mentions:Q'),
                            y=alt.Y('City:N', sort='-x'),
                            tooltip=['City', 'Mentions']
                        ).properties(height=250)
                        st.altair_chart(cities_bar, use_container_width=True)
                    else:
                        st.info("No city data available to plot.")

                st.markdown("---")
                st.markdown("#### 🗃️ Interactive Shipment Ledger Explorer")
                st.markdown("<p style='font-size: 0.85rem; color: #8D99AE;'>Filter and search raw dataset records.</p>", unsafe_allow_html=True)
                
                # Format dataframe for display
                display_df = df[['id', 'timestamp', 'overall_risk', 'has_feedback', 'raw_input_text']].copy()
                display_df['overall_risk'] = display_df['overall_risk'].str.upper()
                display_df.rename(columns={
                    'id': 'Shipment ID', 
                    'timestamp': 'Date/Time',
                    'overall_risk': 'Risk Level',
                    'has_feedback': 'DPO Feedback',
                    'raw_input_text': 'Original Instruction'
                }, inplace=True)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )

        else:
            st.error("Failed to load shipment data for analytics.")
    except Exception as e:
        st.error(f"Failed to render dashboard: {e}")
