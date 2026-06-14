#!/bin/bash
echo "Starting FreightSense AI Engines..."

# Start FastAPI in the background
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} &
FASTAPI_PID=$!

# Start Streamlit
streamlit run app.py --server.port ${STREAMLIT_SERVER_PORT:-8501} --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID
