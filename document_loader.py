import re
import pymupdf  # PyMuPDF
from typing import Tuple

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF.
    Handles encrypted, scanned, corrupted, or empty PDFs gracefully.
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Unable to read PDF file. It might be corrupted or invalid: {str(e)}")

    if doc.is_encrypted:
        try:
            decrypted = doc.authenticate("")
            if not decrypted:
                raise ValueError("The uploaded PDF is password protected. Please provide an unencrypted PDF.")
        except Exception:
            raise ValueError("The uploaded PDF is password protected.")

    text_chunks = []
    has_images = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text and text.strip():
            text_chunks.append(text)
        if len(page.get_images()) > 0:
            has_images = True

    combined_text = "\n".join(text_chunks).strip()

    if not combined_text:
        if has_images:
            raise ValueError(
                "No selectable text found in the PDF. It appears to be a scanned image or screenshot PDF. "
                "Please upload a text-based PDF or a TXT document."
            )
        else:
            raise ValueError("The uploaded PDF does not contain any text content.")

    return combined_text

def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decodes text from TXT bytes, supporting UTF-8 and standard fallback encodings.
    """
    if not file_bytes or not file_bytes.strip():
        raise ValueError("The uploaded TXT file is empty.")

    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            text = file_bytes.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
            
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Could not decode text from the TXT file.")
    return text

def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespace, formatting quotes, and joining hyphenated line breaks.
    """
    if not text:
        return ""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Rejoin hyphenated line breaks (e.g. 'com-\nmunication' -> 'communication')
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)
    # Collapse multiple consecutive newlines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespaces per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join([line for line in lines if line]).strip()

def validate_text(text: str) -> Tuple[bool, str]:
    """
    Checks if text is valid, non-empty, and contains enough words/sentences for FAQ generation.
    """
    if not text or not text.strip():
        return False, "The uploaded document contains no text. Please upload a valid document."

    words = text.split()
    if len(words) < 15:
        return False, f"The document is too brief ({len(words)} words). Please provide a document with at least 15-20 words for meaningful FAQ generation."

    return True, ""
