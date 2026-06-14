# -*- coding: utf-8 -*-
"""
Hugging Face Spaces specific entrypoint for Streamlit.
Simply imports and runs the main app.py.
"""
import sys
import os

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
