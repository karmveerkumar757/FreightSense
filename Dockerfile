# Stage 1: Build system dependencies
FROM python:3.10-slim as builder

# Install system dependencies required for OCR, Audio processing, and compiling certain python packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Install Python dependencies
FROM builder as env_setup

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies. Use --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: App Code
FROM env_setup as final

WORKDIR /app

# Copy the entire project
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# The deployment script will start both services
CMD ["bash", "deploy/start.sh"]
