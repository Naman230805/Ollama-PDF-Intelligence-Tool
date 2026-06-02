"""
core/pdf_extractor.py
Extracts text from PDFs — handles both text-based and scanned (image) PDFs.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path


def extract_text_per_page(pdf_path: str, use_ocr: bool = False) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of dicts: [{page_num, text, is_scanned}]

    Args:
        pdf_path: Path to the PDF file
        use_ocr: If True, attempt OCR on pages with no text (scanned PDFs)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            is_scanned = len(text.strip()) < 30  # Very little text = likely scanned

            if is_scanned and use_ocr:
                text = _ocr_page(pdf_path, i)

            pages.append({
                "page_num": i,
                "text": text.strip(),
                "is_scanned": is_scanned,
                "char_count": len(text.strip()),
            })

    return pages


def extract_full_text(pdf_path: str, use_ocr: bool = False) -> str:
    """Extract all text from PDF as a single string."""
    pages = extract_text_per_page(pdf_path, use_ocr)
    return "\n\n--- PAGE BREAK ---\n\n".join(
        f"[Page {p['page_num']}]\n{p['text']}" for p in pages
    )


def get_page_count(pdf_path: str) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def _ocr_page(pdf_path: str, page_num: int) -> str:
    """
    OCR a single page using PyMuPDF + pytesseract.
    Requires: tesseract installed on system (sudo apt install tesseract-ocr)
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng")
    except ImportError:
        return "[OCR unavailable: install pytesseract and tesseract-ocr]"
    except Exception as e:
        return f"[OCR failed: {e}]"


def get_pdf_metadata(pdf_path: str) -> dict:
    """Get basic PDF metadata."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "pages": doc.page_count,
        "file_size_kb": round(Path(pdf_path).stat().st_size / 1024, 1),
    }
