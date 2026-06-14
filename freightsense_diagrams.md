# FreightSense Architecture Diagrams

Here are the block diagrams that you can use to replace the placeholders in your IEEE LaTeX paper.

> [!TIP]
> You can take a screenshot of these diagrams directly from this interface, or copy the Mermaid code block below each image and paste it into [Mermaid Live Editor](https://mermaid.live/) to export them as high-quality PNG/SVG files for your paper.

---

### Figure 1: Block Diagram of FreightSense Architecture

```mermaid
graph TD
    %% Styling
    classDef default fill:#1E1E2E,stroke:#4D96FF,stroke-width:2px,color:#FFF;
    classDef highlight fill:#FF6B6B,stroke:#FFF,stroke-width:2px,color:#FFF;
    classDef db fill:#6BCB77,stroke:#FFF,stroke-width:2px,color:#000;
    
    %% Input Layer
    U[Fleet Dispatcher] -->|Raw Text / Audio Instruction| UI[Streamlit Frontend Interface]
    
    %% Processing Layer
    subgraph "FreightSense AI Core Backend"
        UI --> NLP[NLP Module: Bi-LSTM-CRF]
        NLP -->|Extracted Entities: Cargo, Locations| RAG[Compliance RAG Engine]
        
        subgraph "Agentic RAG System"
            RAG <-->|Semantic Search| VDB[(ChromaDB: Motor Vehicles Act)]
            RAG <-->|Reasoning & Generation| LLM[Google Gemini LLM API]
        end
        
        NLP -->|Extracted Locations| VRP[Multi-Objective VRP Optimizer]
        RAG -->|Risk Severity Scores| VRP
        
        subgraph "Geospatial Routing"
            VRP <-->|Road Distance Matrix| OSRM[OSRM Route API]
        end
    end
    
    %% Output Layer
    RAG -->|Plain English Advisory| OUT1[Text Advisory Generation]
    VRP -->|Optimized Route & Metrics| OUT2[Geospatial Folium Map]
    OUT1 --> TTS[Bhashini Text-to-Speech]
    TTS -->|Hindi Audio Voice Note| OUT3[Driver Notification System]
    
    OUT2 --> UI
    OUT3 --> UI
    OUT1 --> UI
    
    class VDB db;
    class UI highlight;
```

---

### Figure 2: Step-by-Step Pipeline Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor D as Dispatcher
    participant F as Frontend (Streamlit)
    participant N as NER Extractor (Bi-LSTM-CRF)
    participant R as Agentic RAG (Chroma + Gemini)
    participant V as MO-VRP Engine (OSRM)
    participant B as Bhashini TTS
    
    D->>F: Input mixed Hindi-English dispatch text
    activate F
    F->>N: Send raw text for entity extraction
    activate N
    N-->>F: Return structured JSON (Locations, Cargo, Rules)
    deactivate N
    
    F->>R: Send Entities & Constraints for compliance check
    activate R
    R->>R: Retrieve legal context from Vector Database
    R->>R: Generate Risk Score & Advisory via LLM
    R-->>F: Return Compliance Advisory JSON
    deactivate R
    
    F->>V: Send Locations & Computed Risk Scores
    activate V
    V->>V: Query OSRM & compute optimal Pareto front
    V-->>F: Return Optimized Route (Lat/Lon Coordinates)
    deactivate V
    
    F->>B: Send Advisory Text for regional translation
    activate B
    B-->>F: Return Translated Hindi Audio (.wav)
    deactivate B
    
    F-->>D: Render Interactive Map, Advisory Text, & Audio Player
    deactivate F
```
