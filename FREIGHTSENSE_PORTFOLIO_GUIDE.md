# FreightSense: AI-Powered Logistics Constraint & Compliance Risk Advisory Engine

## 1. Executive Summary & Goal
**FreightSense** is an enterprise-grade, AI-driven logistics and dispatch advisory system. The core goal of this project is to solve a massive pain point in the Indian supply chain industry: **Unstructured Dispatch Communication**.

Fleet managers and dispatchers often send complex, unstructured instructions via text, voice notes, or scanned challans (e.g., *"Urgent medicine delivery Bangalore se Pune before 5 PM. Keep frozen. Avoid toll 4."*). If a driver misses a subtle compliance rule (like a city truck-ban timing, e-way bill expiration, or hazmat permit requirement), it results in heavy fines and delayed shipments.

**FreightSense** acts as an intelligent middleware. It ingests these unstructured multi-modal inputs, extracts critical logistical constraints using custom NLP models, cross-references them against legal regulations using a Hybrid Search RAG (Retrieval-Augmented Generation) pipeline, and generates a dynamic, optimized route and real-time hazard alerts for the driver.

---

## 2. System Architecture & Tech Stack
The project is built on a modern, decoupled architecture designed for scalability and MLOps integration.

### Core Stack
*   **Backend:** FastAPI (Python) for high-performance, asynchronous REST API endpoints.
*   **Frontend:** Streamlit with a custom premium "Glassmorphic" CSS UI and interactive Folium maps.
*   **Database (MLOps):** SQLite for logging shipment history, tracking compliance risks, and capturing dispatcher feedback for continuous model retraining.

### Artificial Intelligence & NLP
*   **Entity Extraction (NER):** Custom fine-tuned `spaCy` model trained to recognize logistics-specific entities (Cargo, Time Constraints, Routes, Handling Rules, Compliance Documents).
*   **Intent Classification:** Fine-tuned `BERT` (Hugging Face) to understand the context of sentences (e.g., recognizing if a sentence is about a route change or a temperature control requirement).
*   **Compliance Verification (RAG):** `ChromaDB` vector database storing Indian Motor Vehicles Act documents and local city truck bans. Enhanced with a **Hybrid Search** engine (Semantic Vector Search + TF-IDF Keyword Re-ranking) for pinpoint legal accuracy.
*   **LLM Advisory:** `Google Gemini API` synthesizes the extracted constraints and retrieved laws to generate a plain-English risk advisory.

### Routing & External APIs
*   **VRP Optimization:** Google `OR-Tools` handles complex Vehicle Routing Problem (VRP) logic to optimize multi-stop delivery sequences based on time and distance matrices.
*   **Distance Matrix:** `OSRM (Open Source Routing Machine)` provides real-world road distances and base travel times.
*   **Predictive Hazards:** `Open-Meteo API` fetches live weather data for route nodes (Rain, Fog, Thunderstorms) to flag driving risks.
*   **Real-time Alerts:** Integrated `Telegram Bot API` (and Twilio structure) for dispatching instant SOS or re-routing alerts to drivers' phones.

---

## 3. End-to-End Workflow (How It Works)

### Phase 1: Multi-Modal Ingestion
The system can accept dispatch instructions via three modes:
1.  **Manual Text:** Mixed Hinglish/English text input.
2.  **Document OCR:** Uploading scanned PDFs or Challans (Processed via `PyMuPDF` + `Tesseract OCR`).
3.  **Voice Notes:** Uploading WhatsApp audio files (Processed via `OpenAI Whisper` for transcription).

### Phase 2: AI Constraint Extraction & RAG
Once the text is normalized, the custom `spaCy` NER model extracts entities like `[Cargo: Medicines]`, `[Location: Pune]`, and `[Handling: Keep Frozen]`. 
Simultaneously, the Hybrid RAG engine searches the local ChromaDB for regulations related to the extracted entities (e.g., searching for "Pune truck entry timings"). 

### Phase 3: Route Optimization & Hazard Detection
The system identifies all geographical nodes in the text. It queries OSRM for distance matrices and feeds them into Google OR-Tools. OR-Tools calculates the most efficient sequence of stops (TSP/VRP). The system then queries Open-Meteo for live weather at each stop and applies predictive traffic heuristics to estimate realistic travel times.

### Phase 4: Advisory Generation & Dashboarding
The Gemini API generates a final JSON advisory classifying the overall risk (LOW, MEDIUM, HIGH). This is displayed on the Streamlit dashboard alongside:
*   A visual map plotting the optimized route, weather hazards, and restricted zones (e.g., Delhi Ring Road).
*   An Executive Analytics Dashboard tracking pipeline drift, total shipments processed, and risk distribution charts.

### Phase 5: Driver Dispatch & MLOps Feedback
The dispatcher can simulate a real-time hazard (like a sudden road closure) and instantly ping the driver's Telegram with a re-routing alert. Finally, if the AI made a mistake, the dispatcher can submit feedback via the UI, which is logged into SQLite. An automated evaluation script (`tests/eval_metrics.py`) calculates F1-scores, allowing the engineering team to monitor Model Drift over time.

---

## 4. Key Highlights for Interviewers
When presenting this project, emphasize the following enterprise-grade engineering practices used:

1.  **Hybrid RAG Search:** Explain that you didn't just use standard semantic search. You built a custom TF-IDF keyword re-ranker on top of ChromaDB to ensure exact legal terms aren't missed.
2.  **MLOps Pipeline & Automated Eval:** Highlight `tests/eval_metrics.py`. Show that you understand machine learning isn't just about training a model once, but evaluating Precision/Recall/F1-score continuously in production.
3.  **Algorithmic Routing (OR-Tools):** Mention that you implemented a mathematical solver for the Vehicle Routing Problem, going beyond simple point A to B APIs.
4.  **Premium UI/UX:** The application doesn't look like a standard data science script. It features a polished, modern glassmorphic interface with an Executive Analytics dashboard built with Pandas.

## 5. Conclusion
FreightSense bridges the gap between chaotic real-world human communication and strict regulatory/logistical constraints. It transforms reactive dispatching into proactive, AI-assisted fleet management.
