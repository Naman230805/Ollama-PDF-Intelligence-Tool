#!/usr/bin/env python3
"""
classify_pdf.py — CLI for classifying PDF pages into user-defined categories.

Usage:
    python classify_pdf.py --pdf document.pdf --categories categories.json
    python classify_pdf.py --pdf doc.pdf --cats "Invoice" "Resume" "Legal" "Receipt"
    python classify_pdf.py --pdf doc.pdf --cats "Invoice" "Resume" --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ollama_client import is_ollama_running, list_models
from classifier import classify_pdf, load_categories_from_file


def progress_bar(current, total):
    pct = int(current / total * 40)
    bar = "█" * pct + "░" * (40 - pct)
    print(f"\r  [{bar}] {current}/{total} pages", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Classify PDF pages into user-defined categories (offline, uses Ollama)"
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument(
        "--categories",
        help="Path to a JSON file with categories list",
    )
    parser.add_argument(
        "--cats",
        nargs="+",
        metavar="CATEGORY",
        help="Category names directly (e.g. --cats Invoice Resume Legal)",
    )
    parser.add_argument("--output", help="Save JSON report to this file")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for scanned PDFs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Pages per LLM call (default: 5). Lower = slower but more accurate.",
    )
    args = parser.parse_args()

    # Resolve categories
    categories = []
    if args.categories:
        categories = load_categories_from_file(args.categories)
    elif args.cats:
        categories = args.cats
    else:
        print("❌ Provide categories via --categories file.json or --cats Cat1 Cat2 ...")
        sys.exit(1)

    print(f"\n📚 Categories ({len(categories)}): {', '.join(categories)}")

    # Pre-flight checks
    print("\n🔍 Checking Ollama server...")
    if not is_ollama_running():
        print("❌ Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    models = list_models()
    if args.model not in models:
        print(f"❌ Model '{args.model}' not found locally.")
        print(f"   Available: {models or ['none']}")
        print(f"   Pull: ollama pull {args.model}")
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"\n📄 PDF: {pdf_path.name}")
    print(f"🤖 Model: {args.model} | Batch size: {args.batch_size} | OCR: {args.ocr}")
    print("\n⏳ Classifying pages...")

    report = classify_pdf(
        str(pdf_path),
        categories=categories,
        model=args.model,
        use_ocr=args.ocr,
        batch_size=args.batch_size,
        on_progress=progress_bar,
    )

    print()  # newline after progress bar
    report.print_summary()

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report.to_json(), encoding="utf-8")
        print(f"💾 Full report saved to: {out_path}\n")


if __name__ == "__main__":
    main()
