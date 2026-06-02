#!/usr/bin/env python3
"""
app.py — Gradio UI. Run: python app.py → http://localhost:7860
"""

import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from ollama_client import is_ollama_running, list_models
from redactor import extract_document_info
from pdf_redactor import redact_pdf
from classifier import classify_pdf

def get_models():
    if not is_ollama_running():
        return ["(Ollama not running — run: ollama serve)"]
    m = list_models()
    return m if m else ["(No models — run: ollama pull llama3.2)"]


# ── Redaction ─────────────────────────────────────────────────────────────────

def run_redaction(pdf_file, model, custom_fields_text, show_label, extract_info, doc_type):
    if pdf_file is None:
        return "❌ Upload a PDF first.", None, ""

    if not is_ollama_running():
        return "❌ Ollama not running — run **ollama serve** in a terminal.", None, ""

    custom_fields = [f.strip() for f in custom_fields_text.strip().splitlines() if f.strip()] \
                    if custom_fields_text.strip() else None

    tmp = tempfile.NamedTemporaryFile(suffix="_redacted.pdf", delete=False)
    tmp.close()

    try:
        summary = redact_pdf(
            pdf_file.name,
            output_path=tmp.name,
            ai_model=model,
            custom_fields_to_hide=custom_fields,
            show_label=show_label,
            document_type=doc_type,
        )
    except Exception as e:
        return f"❌ Error: {e}", None, ""

    status = f"✅ **{summary.total_redactions} item(s) redacted** — Pages affected: {summary.pages_affected or 'none'}\n\n"

    if summary.redactions_by_type:
        status += "**Redacted by type:**\n"
        for t, c in summary.redactions_by_type.items():
            status += f"- {t}: {c}\n"

    if summary.ai_findings:
        status += f"\n**AI findings ({len(summary.ai_findings)}):**\n"
        for f in summary.ai_findings[:15]:
            status += f"- [{f['type']}]: {f['reason']}\n"

    if not summary.total_redactions:
        status += "\n_No PII detected. If this is a scanned PDF, OCR support is needed._"

    extracted_json = ""
    if extract_info:
        try:
            result = extract_document_info(pdf_file.name, model=model, document_type=doc_type)
            extracted_json = json.dumps(result.extracted_info, indent=2, ensure_ascii=False)
        except Exception as e:
            extracted_json = f"// Extraction failed: {e}"

    return status, tmp.name, extracted_json


# ── Classifier ────────────────────────────────────────────────────────────────

def run_classification(pdf_file, model, categories_text, batch_size):
    if pdf_file is None:
        return "❌ Upload a PDF first.", "", ""
    if not is_ollama_running():
        return "❌ Ollama not running — run **ollama serve**", "", ""

    categories = [c.strip() for c in categories_text.strip().splitlines() if c.strip()]
    if len(categories) < 2:
        return "❌ Enter at least 2 categories.", "", ""

    try:
        report = classify_pdf(pdf_file.name, categories=categories, model=model, batch_size=int(batch_size))

        summary_lines = ["| Category | Count | Pages |", "|----------|-------|-------|"]
        for cat, pages in sorted(report.category_summary.items()):
            summary_lines.append(f"| {cat} | {len(pages)} | {pages} |")

        detail_lines = ["\n### Per-Page Detail\n",
                        "| Page | Category | Confidence | Reason |",
                        "|------|----------|------------|--------|"]
        for p in report.per_page:
            icon = {"high": "✅", "medium": "⚠️", "low": "❓"}.get(p.confidence, "")
            detail_lines.append(f"| {p.page_num} | {p.category} | {icon} {p.confidence} | {p.reason} |")

        status = f"✅ **{len(report.per_page)} pages** classified into **{len(report.category_summary)} categories**"
        return status, "\n".join(summary_lines + detail_lines), report.to_json()
    except Exception as e:
        return f"❌ Error: {e}", "", ""


# ── UI ────────────────────────────────────────────────────────────────────────

DEFAULT_CATS = "\n".join([
    "Invoice", "Purchase Order", "Resume / CV", "Legal Contract",
    "Bank Statement", "Medical Report", "Academic Marksheet", "Tax Document",
    "Government ID", "Insurance Policy", "Project Report", "Meeting Minutes",
    "Technical Specification", "Email Printout", "Receipt",
    "Warranty Document", "Admission Letter", "Non-Disclosure Agreement",
    "Property Document", "Unclassified"
])

with gr.Blocks(title="Ollama PDF Intelligence Tool", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🧠 Ollama PDF Intelligence Tool — Offline & Private
> **Start Ollama first:** `ollama serve` | **Pull model:** `ollama pull llama3.2`
""")

    with gr.Row():
        model_dd = gr.Dropdown(choices=get_models(), label="🤖 Ollama Model", scale=3)
        gr.Button("🔄 Refresh", scale=1).click(
            fn=lambda: gr.update(choices=get_models()), outputs=model_dd
        )

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1: Redact ──────────────────────────────────────────────────
        with gr.Tab("🔒 Redact PDF"):
            gr.Markdown("""
### AI reads your document and hides all sensitive information
The model automatically detects Aadhaar, PAN, phone, email, DOB, bank details, names, addresses — anything it understands as private.
You can also tell it to hide additional specific fields.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_in1 = gr.File(label="📤 Upload PDF", file_types=[".pdf"])

                    gr.Markdown("#### 📄 Document Type")
                    gr.Markdown("*Tell the AI what kind of document this is so it knows what's safe to keep vs hide.*")
                    doc_type = gr.Radio(
                        ["auto", "resume", "marksheet", "medical", "invoice"],
                        value="auto",
                        label="Document type",
                    )

                    gr.Markdown("#### 🟡 Custom Fields to Hide *(optional)*")
                    gr.Markdown("Type any extra field names you want hidden — one per line. The AI will find and black out their values.")
                    custom_fields = gr.Textbox(
                        label="Additional fields to hide",
                        placeholder="roll number\nfather name\nmother name\nguardian address\nstudent ID",
                        lines=5,
                    )

                    gr.Markdown("#### ⚙️ Options")
                    show_label   = gr.Checkbox(label='Show field type on black bars', value=True)
                    extract_info = gr.Checkbox(label="Also extract clean info (name, marks, etc.)", value=False)

                    btn1 = gr.Button("🚀 Redact PDF", variant="primary", size="lg")

                with gr.Column(scale=2):
                    status1   = gr.Markdown("Upload a PDF and click Redact.")
                    pdf_out   = gr.File(label="⬇️ Download Redacted PDF", interactive=False)
                    extracted = gr.Code(label="📋 Extracted Info (JSON)", language="json", lines=12)

            btn1.click(
                fn=run_redaction,
                inputs=[pdf_in1, model_dd, custom_fields, show_label, extract_info, doc_type],
                outputs=[status1, pdf_out, extracted],
            )

        # ── Tab 2: Classify ────────────────────────────────────────────────
        with gr.Tab("📂 Classify Pages"):
            gr.Markdown("### Classify every page into your custom categories")
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_in2    = gr.File(label="📤 Upload PDF", file_types=[".pdf"])
                    cats_input = gr.Textbox(label="📚 Categories (one per line)",
                                            value=DEFAULT_CATS, lines=14)
                    batch_size = gr.Slider(1, 10, value=5, step=1, label="Pages per LLM call")
                    btn2       = gr.Button("🚀 Classify Pages", variant="primary", size="lg")
                with gr.Column(scale=2):
                    status2      = gr.Markdown("Upload a PDF and click Classify.")
                    results_md   = gr.Markdown()
                    results_json = gr.Code(label="Full JSON Report", language="json", lines=12)

            btn2.click(
                fn=run_classification,
                inputs=[pdf_in2, model_dd, cats_input, batch_size],
                outputs=[status2, results_md, results_json],
            )

        # ── Tab 3: Setup ───────────────────────────────────────────────────
        with gr.Tab("⚙️ Setup & CLI"):
            gr.Markdown("""
## Setup (one-time)
```bash
curl -fsSL https://ollama.com/install.sh | sh   # Install Ollama
ollama pull llama3.2                             # Download model (~2GB)
pip install -r requirements.txt                 # Python deps
```

## Run
```bash
ollama serve       # Terminal 1 — keep open
python app.py      # Terminal 2 — open http://localhost:7860
```

---

## CLI Examples

```bash
# AI detects all PII automatically
python redact_pdf.py --pdf marksheet.pdf

# AI + also hide specific fields
python redact_pdf.py --pdf marksheet.pdf --hide "roll number" "father name"

# Redact + save extracted clean info
python redact_pdf.py --pdf marksheet.pdf --extract-info --output-json info.json

# Plain black bars (no labels)
python redact_pdf.py --pdf marksheet.pdf --no-label
```

---

## How it works
1. Ollama model reads each page of your PDF
2. It identifies everything that looks like personal/sensitive information
3. The exact values are located on the page and covered with permanent black bars
4. You download the redacted PDF — the original text under the bars is destroyed, not just hidden
""")

if __name__ == "__main__":
    print("\n🚀 Starting... open http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
