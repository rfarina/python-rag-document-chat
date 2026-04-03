from PyPDF2 import PdfReader

def pdf_to_text(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")
