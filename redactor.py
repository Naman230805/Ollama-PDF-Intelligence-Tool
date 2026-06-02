"""
core/redactor.py
Feature 1: Extract useful info from documents while hiding PII.
Works great for: marksheets, certificates, ID cards, admit cards, etc.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional
from ollama_client import ask_json
from pdf_extractor import extract_full_text, get_pdf_metadata


# ─── Regex-based pre-redaction (fast, runs before LLM) ───────────────────────

PII_PATTERNS = {
    "aadhaar_number": [
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",          # 12-digit with optional spaces
        r"\b\d{4}-\d{4}-\d{4}\b",               # with dashes
    ],
    "pan_number": [
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    ],
    "phone_number": [
        r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",       # Indian mobile
        r"\b0\d{2,4}[-\s]?\d{6,8}\b",           # Landline
    ],
    "email": [
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    ],
    "date_of_birth": [
        r"\bD\.?O\.?B\.?\s*:?\s*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
        r"\bDate of Birth\s*:?\s*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
    ],
    "bank_account": [
        r"\b\d{9,18}\b(?=.*(?:account|a\/c|acct))",  # Near account keywords
    ],
    "ifsc_code": [
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    ],
    "passport_number": [
        r"\b[A-Z][1-9][0-9]{7}\b",
    ],
}

REDACT_PLACEHOLDER = {
    "aadhaar_number": "[AADHAAR REDACTED]",
    "pan_number": "[PAN REDACTED]",
    "phone_number": "[PHONE REDACTED]",
    "email": "[EMAIL REDACTED]",
    "date_of_birth": "[DOB REDACTED]",
    "bank_account": "[ACCOUNT REDACTED]",
    "ifsc_code": "[IFSC REDACTED]",
    "passport_number": "[PASSPORT REDACTED]",
}


@dataclass
class RedactionResult:
    original_text: str
    redacted_text: str
    found_pii_types: list[str]
    extracted_info: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "extracted_info": self.extracted_info,
            "redacted_fields": self.found_pii_types,
            "redacted_text": self.redacted_text,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def regex_redact(text: str) -> tuple[str, list[str]]:
    """
    Fast regex-based PII removal. Returns (redacted_text, list_of_found_types).
    This runs BEFORE the LLM for guaranteed redaction even if LLM fails.
    """
    found = []
    for pii_type, patterns in PII_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(
                    pattern,
                    REDACT_PLACEHOLDER[pii_type],
                    text,
                    flags=re.IGNORECASE,
                )
                if pii_type not in found:
                    found.append(pii_type)
    return text, found


def extract_document_info(
    pdf_path: str,
    model: str = "llama3.2",
    document_type: str = "auto",
    use_ocr: bool = False,
    custom_fields: Optional[list[str]] = None,
) -> RedactionResult:
    """
    Main function: Extract structured info from a document while redacting PII.

    Args:
        pdf_path: Path to the PDF
        model: Ollama model name
        document_type: "marksheet", "certificate", "id_card", "invoice", or "auto"
        use_ocr: Enable OCR for scanned PDFs
        custom_fields: Extra fields to extract (e.g. ["roll_number", "grade"])
    """
    # Step 1: Extract raw text
    raw_text = extract_full_text(pdf_path, use_ocr=use_ocr)
    meta = get_pdf_metadata(pdf_path)

    # Step 2: Regex redaction (always runs, fast & reliable)
    redacted_text, found_pii = regex_redact(raw_text)

    # Step 3: LLM extracts structured info from the already-redacted text
    system_prompt = _build_system_prompt(document_type, custom_fields)
    user_prompt = _build_extraction_prompt(redacted_text, document_type)

    extracted = ask_json(user_prompt, model=model, system=system_prompt)

    # Step 4: If LLM found additional PII it wants to flag, merge it
    llm_pii = extracted.pop("additional_pii_found", [])
    if isinstance(llm_pii, list):
        for item in llm_pii:
            if item not in found_pii:
                found_pii.append(item)

    return RedactionResult(
        original_text=raw_text,
        redacted_text=redacted_text,
        found_pii_types=found_pii,
        extracted_info=extracted,
        metadata=meta,
    )


# ─── Prompt builders ──────────────────────────────────────────────────────────

def _build_system_prompt(document_type: str, custom_fields: Optional[list]) -> str:
    base = """You are a document information extractor. Your job is to:
1. Extract ONLY the useful, non-sensitive information from documents
2. NEVER include Aadhaar numbers, PAN numbers, phone numbers, emails, or dates of birth in your output
3. Return structured JSON only

Sensitive data will already be replaced with [REDACTED] tags in the input — do NOT un-redact them."""

    type_hints = {
        "marksheet": """
For marksheets/mark statements, extract:
- student_name, father_name (optional), mother_name (optional)
- school_name / college_name
- board / university
- exam_year, exam_month
- class / standard (e.g. "12th", "10th")
- roll_number (if present — this is NOT PII, it's a public identifier)
- subjects: {subject_name: marks_obtained} 
- total_marks, marks_obtained, percentage
- result (Pass/Fail)
- division / grade / rank (if present)
- additional_pii_found: list any other PII you see that wasn't redacted""",

        "certificate": """
For certificates, extract:
- recipient_name
- certificate_type
- issuing_organization
- issue_date (year only is fine)
- course_name / achievement
- valid_until (if applicable)
- additional_pii_found: []""",

        "id_card": """
For ID cards, extract:
- name
- organization / institution
- designation / role / department
- id_number (employee/student ID — NOT Aadhaar/PAN)
- validity
- additional_pii_found: []""",

        "invoice": """
For invoices, extract:
- vendor_name, customer_name
- invoice_number, invoice_date
- items: [{description, quantity, rate, amount}]
- subtotal, tax, total_amount
- payment_terms
- additional_pii_found: []""",

        "auto": """
Detect the document type first, then extract the most relevant non-sensitive fields.
Include a "document_type" field in your response.
additional_pii_found: list any PII you see."""
    }

    hint = type_hints.get(document_type, type_hints["auto"])
    result = base + "\n\n" + hint

    if custom_fields:
        result += f"\n\nAlso extract these additional fields if present: {', '.join(custom_fields)}"

    return result


def _build_extraction_prompt(redacted_text: str, document_type: str) -> str:
    # Truncate very long texts to avoid context overflow
    max_chars = 6000
    if len(redacted_text) > max_chars:
        redacted_text = redacted_text[:max_chars] + "\n... [truncated]"

    return f"""Extract structured information from this {document_type} document.
The text has already been pre-processed to replace sensitive data with [REDACTED] tags.

DOCUMENT TEXT:
{redacted_text}

Return a JSON object with the extracted fields. Do not include any sensitive/PII data."""
