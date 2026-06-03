# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.genai.rag_retriever import ingest_regulations_folder
from src.feedback.feedback_db import init_db

print("🚀 Starting FreightSense System Ingestion and Initialization...")

# 1. Initialize SQLite Database
print("\n🗄️ Initializing SQLite Local Database...")
init_db()
print("✅ SQLite database initialized.")

# 2. Ingest Regulations into ChromaDB
print("\n📚 Indexing Regulations into ChromaDB Vector Store...")
ingest_regulations_folder()
print("✅ ChromaDB Indexing Complete.")

print("\n🎉 FreightSense Initial Setup Done Successfully!")
