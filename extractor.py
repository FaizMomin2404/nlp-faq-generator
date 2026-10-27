import re
import fitz  # PyMuPDF
from typing import Tuple

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF (fitz).
    """
    text_chunks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()

def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decodes text from TXT bytes, supporting utf-8 and fallback encodings.
    """
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore").strip()

def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespaces, formatting quotes, and removing control chars.
    """
    if not text:
        return ""
    # Normalize multiple newlines and carriage returns
    text = text.replace("\r", " ")
    # Replace non-breaking spaces and irregular whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Rejoin hyphenated line breaks (e.g., 'arti-\nficial' -> 'artificial')
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Normalize multiple line breaks to standard sentence separators
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def validate_text(text: str) -> Tuple[bool, str]:
    """
    Checks if text is valid, non-empty, and sufficiently long for FAQ generation.
    """
    if not text or not text.strip():
        return False, "The uploaded document is empty or text could not be extracted."
    
    words = text.split()
    if len(words) < 15:
        return False, f"The document is too short ({len(words)} words). Please provide at least 15-20 words for meaningful FAQ generation."
    
    return True, ""
