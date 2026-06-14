# 🚛 FreightSense
**AI-Powered Logistics Constraint & Compliance Risk Advisory Engine (v5.0)**

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)

FreightSense is an end-to-end AI advisory engine designed for the logistics and supply chain industry in India. It bridges the gap between unstructured, code-mixed dispatcher instructions (Hindi-English) and strict regulatory compliance by utilizing advanced Natural Language Processing (NLP) and Operations Research (OR) techniques.

## ✨ Key Features
- **Code-Mixed NER Extractor:** Utilizes a custom-trained **Bi-LSTM-CRF** model to accurately extract entities (`Locations`, `Cargo_Type`, `Constraints`) from unstructured "Hinglish" instructions or voice notes.
- **Agentic RAG Engine:** Queries Indian Motor Vehicles Act rules from a **ChromaDB** vector store and generates plain-English risk assessments using the Google **Gemini LLM**.
- **Multi-Objective VRP:** Generates the optimal route by balancing distance (via **OSRM API**), compliance risk, and probabilistic delays. Renders an interactive **Folium Map**.
- **Multilingual Audio Advisory:** Translates and speaks the advisory in Hindi (or other regional languages) using the **Bhashini TTS API**.
- **Executive Analytics Dashboard:** Provides an interactive **Altair** dashboard for risk distributions, shipment volume tracking, and data exploration.

## 🏗️ System Architecture
*(Insert architecture diagram here)*

## 🚀 Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/karmveerkumar757/FreightSense.git
cd FreightSense
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 💻 Usage

To run the full Streamlit web application:

```bash
# On Windows, you can simply double-click or run:
.\run_freightsense.bat

# Alternatively, run via Streamlit:
streamlit run app.py
```

## 📂 Directory Structure
- `app.py`: Main Streamlit application frontend.
- `main.py`: FastAPI backend entry point (optional, for decoupling backend).
- `src/`: Core AI modules.
  - `nlp/`: Bi-LSTM-CRF model and Delay Predictor.
  - `genai/`: Agentic RAG and Multi-Objective VRP algorithms.
  - `output/`: Bhashini TTS integration.
- `training/`: Training scripts and Jupyter notebooks for fine-tuning the models.
- `freightsense_ieee_paper.tex`: Complete IEEE-format research paper detailing the methodology and mathematical formulations of this project.

## 🤝 Contribution
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.

---
**Author:** Karmveer  
**License:** MIT
