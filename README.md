# 🚛 FreightSense: AI Logistics Advisory & Constraint Engine

FreightSense is a premium, AI-powered logistics constraint and compliance risk advisory system. It is designed to ingest unstructured dispatcher instructions, freight challans, and voice notes (in English or mixed Hinglish) and generate structured compliance advisories, flag route violations, and dispatch alerts to drivers in real-time.

---

## 🌟 Key Features

1. **Multi-Modal Ingestion:**
   * **Manual Entry:** Direct Hinglish/English text dispatch instructions.
   * **Document Parsing (OCR):** Extracts text from scanned freight challans/receipts using PyMuPDF and Tesseract OCR.
   * **Voice Notes (ASR):** Transcribes Hinglish audio notes using OpenAI Whisper.
2. **NLP & Constraint Extraction:**
   * **Named Entity Recognition (NER):** Extracts key logistics details (Cargo, Time Constraints, Route Nodes, Handling Instructions).
   * **Intent Classification:** Classifies instruction intent using fine-tuned BERT and heuristic classifiers.
3. **Regulations RAG (Retrieval-Augmented Generation):**
   * Semantically searches a **ChromaDB** vector store containing Indian Transport Regulations (e.g., Motor Vehicles Act, local city truck timing restrictions like the Delhi Ring Road ban).
4. **Smart Advisory Generation:**
   * Integrates NLP constraints and regulatory context to compile a plain-English safety, route, and permit advisory.
5. **Driver Alerts:**
   * Dispatches formatted alerts to drivers via **Twilio (WhatsApp/SMS)** or **Telegram**.
6. **MLOps Feedback Loop:**
   * Logs dispatcher feedback and corrected labels into a local database for model drift monitoring and retraining.

---

## 📁 Repository Structure

```text
FreightSense_Project/
├── app.py                     # Streamlit Frontend Dashboard UI
├── main.py                    # FastAPI Backend API Server
├── run_freightsense.bat       # Launcher script for starting both services
├── requirements.txt           # Python project dependencies
├── data/                      # Data files & regulations
│   ├── Indian_Freight_Delivery_Instructions_Master_400.csv  # Dataset
│   ├── regulations/           # Raw text regulation logs for Chroma RAG
│   └── uploads/               # Temp storage for OCR/ASR uploads (git-ignored)
├── vectordb/                  # Vector Database scripts & DB files
│   ├── ingest_data.py         # Script to populate ChromaDB from regulations/
│   └── chroma_db/             # Chroma database files (git-ignored)
├── training/                  # Model training scripts
│   ├── generate_intent_data.py
│   └── finetune_bert.py
├── src/                       # Main application source code
│   ├── ingestion/             # Text cleaning, OCR, and Whisper ASR
│   ├── nlp/                   # NER Extraction and Intent Classification
│   ├── genai/                 # RAG Retriever, Prompt Builder, and LLM Caller
│   ├── output/                # JSON validators and driver notification senders
│   └── feedback/              # SQLite database schemas for logs & MLOps
└── models/                    # Model weights directory (git-ignored)
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have the following installed on your system:
* **Python 3.10+**
* **Tesseract OCR** (added to your system PATH)
* **Git**

### 2. Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/karmveerkumar757/FreightSense.git
   cd FreightSense
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the root directory (based on `.env.example`):
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   TWILIO_ACCOUNT_SID=your_twilio_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_number
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ```

5. Ingest Regulatory Documents into ChromaDB:
   ```bash
   python vectordb/ingest_data.py
   ```

---

## 🚀 Running the Application

You can start both the **FastAPI Backend** and the **Streamlit Dashboard** using the helper script:

```bash
# On Windows:
.\run_freightsense.bat
```

Alternatively, you can start them manually in separate terminal tabs:

* **Start FastAPI Backend (Port 8000):**
  ```bash
  python main.py
  ```
* **Start Streamlit Dashboard (Port 8501):**
  ```bash
  streamlit run app.py
  ```

---

## 🛠️ Technology Stack
* **Frontend:** Streamlit, Folium (interactive maps), CSS (Dark glassmorphism theme)
* **Backend:** FastAPI, Uvicorn, SQLite
* **NLP:** spaCy (NER), Hugging Face Transformers (BERT classification)
* **Generative AI:** Google Gemini API (Advisory compilation), ChromaDB (Vector DB)
* **Audio & Document Ingestion:** OpenAI Whisper, PyMuPDF, Tesseract OCR
* **Integrations:** Twilio API, Telegram Bot API
