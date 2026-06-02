#!/usr/bin/env python3
"""
redact_pdf.py — CLI for AI-powered PDF redaction.

Examples:
  # Standard — AI finds all PII automatically
  python redact_pdf.py --pdf marksheet.pdf

  # Also hide specific custom fields
  python redact_pdf.py --pdf marksheet.pdf --hide "roll number" "father name"

  # Redact + extract clean info as JSON
  python redact_pdf.py --pdf marksheet.pdf --extract-info --output-json result.json
"""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdf_redactor import redact_pdf
from ollama_client import is_ollama_running, list_models
from redactor import extract_document_info


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered PDF redaction — hides PII using Ollama (offline)"
    )
    parser.add_argument("--pdf",    required=True, help="Input PDF path")
    parser.add_argument("--output", help="Output redacted PDF (default: input_redacted.pdf)")
    parser.add_argument("--model",  default="llama3.2", help="Ollama model (default: llama3.2)")
    parser.add_argument("--hide",   nargs="+", metavar="FIELD",
                        help='Extra fields to hide, e.g. --hide "roll number" "father name"')
    parser.add_argument("--no-label",    action="store_true", help="Plain black bars, no labels")
    parser.add_argument("--extract-info",action="store_true", help="Also extract clean info as JSON")
    parser.add_argument("--output-json", help="Save extracted info to this JSON file")
    parser.add_argument("--type", default="auto",
                        choices=["auto","marksheet","certificate","id_card","invoice"],
                        help="Document type for info extraction")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}"); sys.exit(1)

    if not is_ollama_running():
        print("❌ Ollama not running. Start it with: ollama serve"); sys.exit(1)

    if args.model not in list_models():
        print(f"❌ Model '{args.model}' not found. Run: ollama pull {args.model}"); sys.exit(1)

    out_pdf = args.output or str(pdf_path.parent / f"{pdf_path.stem}_redacted.pdf")

    print(f"\n📄 Input  : {pdf_path.name}")
    print(f"💾 Output : {out_pdf}")
    print(f"🤖 Model  : {args.model}")
    if args.hide:
        print(f"🟡 Custom fields to hide: {args.hide}")
    print("\n⏳ AI is reading the document and detecting PII...")

    summary = redact_pdf(
        str(pdf_path),
        output_path=out_pdf,
        ai_model=args.model,
        custom_fields_to_hide=args.hide,
        show_label=not args.no_label,
    )

    print(f"\n{'='*55}")
    print(f"✅ REDACTION COMPLETE")
    print(f"{'='*55}")
    print(f"   Total redacted : {summary.total_redactions} item(s)")
    print(f"   Pages affected : {summary.pages_affected or 'none'}")

    if summary.redactions_by_type:
        print(f"\n   By type:")
        for t, c in summary.redactions_by_type.items():
            print(f"     • {t}: {c}")

    if summary.ai_findings:
        print(f"\n   AI findings ({len(summary.ai_findings)}):")
        for f in summary.ai_findings:
            print(f"     • [{f['type']}] — {f['reason']}")

    print(f"\n💾 Saved: {out_pdf}")

    if args.extract_info:
        print(f"\n🤖 Extracting structured info...")
        result = extract_document_info(str(pdf_path), model=args.model, document_type=args.type)
        print("\n📋 Extracted Info:")
        print(json.dumps(result.extracted_info, indent=2, ensure_ascii=False))
        if args.output_json:
            Path(args.output_json).write_text(result.to_json(), encoding="utf-8")
            print(f"\n💾 JSON saved: {args.output_json}")


if __name__ == "__main__":
    main()
