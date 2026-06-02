"""
core/ai_pii_detector.py

AI-powered PII detection that understands document context.
What counts as "sensitive" depends on the type of document:
  - Resume: name/email/LinkedIn are intentional — only hide Aadhaar, PAN, bank details etc.
  - Marksheet: hide Aadhaar, DOB, phone — keep name, marks, school
  - Medical report: hide almost everything personal
  - etc.
"""

import json
from ollama_client import ask_json


# ── Per-document-type rules ───────────────────────────────────────────────────

DOCUMENT_RULES = {
    "resume": {
        "description": "A resume/CV is a document intentionally shared with employers.",
        "always_hide": [
            "Aadhaar number", "PAN number", "Passport number", "Voter ID",
            "bank account numbers", "IFSC codes", "credit/debit card numbers",
            "passwords", "PINs", "date of birth", "blood group",
            "father's name", "mother's name", "marital status",
        ],
        "never_hide": [
            "candidate's own name", "job title", "professional email address",
            "LinkedIn URL", "GitHub URL", "portfolio URL", "city or general location",
            "university name", "degree", "company names", "job experience",
            "skills", "certifications", "languages",
        ],
    },
    "marksheet": {
        "description": "An academic marksheet issued by a school or board.",
        "always_hide": [
            "Aadhaar number", "PAN number", "phone number", "mobile number",
            "email address", "date of birth", "father's name", "mother's name",
            "home address", "bank details",
        ],
        "never_hide": [
            "student name", "roll number", "registration number", "school name",
            "board name", "exam year", "subject names", "marks", "grades",
            "percentage", "result (pass/fail)",
        ],
    },
    "medical": {
        "description": "A medical report or prescription.",
        "always_hide": [
            "patient name", "Aadhaar number", "PAN number", "phone number",
            "email address", "date of birth", "home address", "blood group",
            "diagnosis", "prescription details", "doctor's registration number",
            "insurance policy number", "bank details",
        ],
        "never_hide": [
            "hospital name", "department name", "general medical terms",
        ],
    },
    "invoice": {
        "description": "A billing invoice or receipt.",
        "always_hide": [
            "bank account numbers", "credit/debit card numbers", "IFSC codes",
            "Aadhaar number", "PAN number of individual (not business)",
            "personal phone number", "personal email",
        ],
        "never_hide": [
            "business name", "GST number", "invoice number", "business address",
            "business PAN", "item descriptions", "amounts", "dates",
        ],
    },
    "auto": {
        "description": "Unknown document type — apply general PII rules.",
        "always_hide": [
            "Aadhaar number", "PAN number", "Passport number", "Voter ID",
            "phone numbers", "email addresses", "bank account numbers",
            "credit/debit card numbers", "IFSC codes", "date of birth",
            "home address", "passwords", "PINs", "blood group",
        ],
        "never_hide": [
            "organisation names", "public figures' names in news context",
            "general location names (city, state, country)",
        ],
    },
}


def detect_pii_with_ai(
    page_text: str,
    model: str = "llama3.2",
    custom_fields_to_hide: list[str] | None = None,
    document_type: str = "auto",
) -> list[dict]:
    """
    Detect PII on a page with awareness of what document type it is.

    Args:
        page_text:             Extracted text of one PDF page
        model:                 Ollama model name
        custom_fields_to_hide: Additional fields the user wants hidden
        document_type:         "resume", "marksheet", "medical", "invoice", or "auto"

    Returns:
        [{"text": "exact value", "type": "PII category", "reason": "why"}, ...]
    """
    if not page_text.strip():
        return []

    rules = DOCUMENT_RULES.get(document_type, DOCUMENT_RULES["auto"])

    custom_section = ""
    if custom_fields_to_hide:
        custom_section = f"""
The user has also specifically requested these fields to be hidden:
{json.dumps(custom_fields_to_hide, indent=2)}
Find the actual values of these fields in the document and include them in your findings,
even if they would not normally be considered sensitive for this document type.
"""

    system_prompt = f"""You are a privacy expert. You are analyzing a {document_type.upper()} document.

DOCUMENT CONTEXT: {rules['description']}

ALWAYS HIDE these types of information:
{json.dumps(rules['always_hide'], indent=2)}

NEVER HIDE these — they are expected and intentional in this document type:
{json.dumps(rules['never_hide'], indent=2)}

RULES:
1. Return the EXACT text string as it appears in the document — do not paraphrase
2. Only flag items from the ALWAYS HIDE list (or user-specified custom fields)
3. Do NOT flag anything from the NEVER HIDE list under any circumstances
4. Each finding must be unique — do not repeat the same value twice
5. Respond ONLY with valid JSON, nothing else"""

    prompt = f"""Analyze this document page and identify sensitive information that should be redacted.

DOCUMENT TEXT:
{page_text[:5000]}
{custom_section}

Return this exact JSON:
{{
  "findings": [
    {{
      "text": "exact text as it appears in the document",
      "type": "category name",
      "reason": "one sentence why this should be hidden in this document type"
    }}
  ]
}}

If nothing needs redacting, return: {{"findings": []}}"""

    result = ask_json(prompt, model=model, system=system_prompt)

    findings = result.get("findings", [])
    validated = []
    for f in findings:
        if isinstance(f, dict) and f.get("text") and f.get("type"):
            validated.append({
                "text":   str(f["text"]).strip(),
                "type":   str(f["type"]).strip(),
                "reason": str(f.get("reason", "")).strip(),
            })
    return validated
