# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Proactively configure Windows Tesseract path if not present in the system PATH
if sys.platform.startswith("win"):
    # Standard installation paths for Tesseract OCR on Windows
    standard_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
    ]
    # Check if tesseract is already available in PATH
    import shutil
    if not shutil.which("tesseract"):
        for path in standard_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"📦 Configured Tesseract path: {path}")
                break

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file. Attempts digital text extraction first.
    If no text is found, falls back to OCR using pytesseract on page images.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    text = ""
    
    # 1. Try digital text extraction
    for page in doc:
        text += page.get_text()
        
    # 2. Fall back to OCR if the extracted text is empty or very short (scanned PDF)
    if len(text.strip()) < 20:
        print("⚠️ Digital extraction returned minimal text. Falling back to OCR...")
        ocr_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to an image
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            try:
                page_text = pytesseract.image_to_string(img)
                ocr_text += page_text + "\n"
            except Exception as e:
                print(f"⚠️ OCR failed on page {page_num+1}: {e}")
                print("Make sure Tesseract OCR is installed and added to your system PATH.")
                break
        if ocr_text.strip():
            text = ocr_text
            
    doc.close()
    return text.strip()

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image using pytesseract.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
        
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        raise RuntimeError(f"OCR Image extraction failed: {e}. Check if Tesseract OCR is installed.")

if __name__ == "__main__":
    print("\n🔍 Testing ocr.py Configuration...")
    import shutil
    has_tesseract = shutil.which("tesseract") is not None or pytesseract.pytesseract.tesseract_cmd != "tesseract"
    print(f"  Tesseract Available: {has_tesseract}")
    if has_tesseract:
        print(f"  Tesseract Command: {pytesseract.pytesseract.tesseract_cmd}")
    print("✅ ocr.py compiled and imported successfully!")
