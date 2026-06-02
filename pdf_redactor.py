"""
core/pdf_redactor.py

Redacts a PDF using AI-powered PII detection.
Two modes that work together:

  Mode 1 — AI Detection : Ollama reads and understands the document,
                          finds all PII intelligently
  Mode 2 — Custom Fields: User specifies extra fields to hide
                          (e.g. "roll number", "father name")
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field


BAR_COLOR   = (0, 0, 0)  # black
LABEL_COLOR = (1, 1, 1)  # white text on bar


@dataclass
class RedactionSummary:
    output_path: str
    total_redactions: int
    redactions_by_type: dict
    pages_affected: list[int]
    ai_findings: list[dict] = field(default_factory=list)


def redact_pdf(
    input_path: str,
    output_path: str | None = None,
    ai_model: str = "llama3.2",
    custom_fields_to_hide: list[str] | None = None,
    show_label: bool = True,
    document_type: str = "auto",
) -> RedactionSummary:
    """
    Redact a PDF using AI + optional custom fields.

    Args:
        input_path:            Source PDF path
        output_path:           Where to save the redacted PDF
        ai_model:              Ollama model name
        custom_fields_to_hide: Extra field names to hide, e.g. ["roll number", "father name"]
        show_label:            Print the PII type on the black bar
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_path}")

    if output_path is None:
        output_path = str(input_path.parent / f"{input_path.stem}_redacted.pdf")

    from ai_pii_detector import detect_pii_with_ai
    doc = fitz.open(str(input_path))
    redactions_by_type: dict[str, int] = {}
    pages_affected: set[int] = set()
    total = 0
    all_findings: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        if not page_text:
            continue  # scanned page — no text layer to redact

        # AI reads the page and returns every PII item it finds
        findings = detect_pii_with_ai(
            page_text,
            model=ai_model,
            custom_fields_to_hide=custom_fields_to_hide,
            document_type=document_type,
        )
        all_findings.extend(findings)

        page_had_redaction = False
        for finding in findings:
            rects = _find_text_on_page(page, finding["text"])
            for rect in rects:
                _draw_bar(page, rect, finding["type"] if show_label else None)
                redactions_by_type[finding["type"]] = redactions_by_type.get(finding["type"], 0) + 1
                total += 1
                page_had_redaction = True

        if page_had_redaction:
            pages_affected.add(page_num)

    for page in doc:
        page.apply_redactions()

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    return RedactionSummary(
        output_path=output_path,
        total_redactions=total,
        redactions_by_type=redactions_by_type,
        pages_affected=sorted(pages_affected),
        ai_findings=all_findings,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_text_on_page(page: fitz.Page, text: str) -> list[fitz.Rect]:
    """Find exact text on a page. Falls back to searching just the value part."""
    if not text or len(text.strip()) < 2:
        return []

    rects = page.search_for(text.strip())
    if rects:
        return [r + fitz.Rect(-2, -1, 2, 1) for r in rects]

    # If AI returned "Label: value", try searching just the value
    if ":" in text:
        value = text.split(":", 1)[-1].strip()
        rects = page.search_for(value)
        if rects:
            return [r + fitz.Rect(-2, -1, 2, 1) for r in rects]

    return []


def _draw_bar(page: fitz.Page, rect: fitz.Rect, label: str | None):
    annot = page.add_redact_annot(
        rect,
        text=label or "",
        fontsize=6,
        fontname="Helv",
        text_color=LABEL_COLOR,
        fill=BAR_COLOR,
        cross_out=False,
    )
    annot.update()
