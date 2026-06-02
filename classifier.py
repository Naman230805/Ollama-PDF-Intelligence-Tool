"""
core/classifier.py
Feature 2: Classify each page of a PDF into user-defined categories.
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from ollama_client import ask_json
from pdf_extractor import extract_text_per_page, get_pdf_metadata


@dataclass
class PageResult:
    page_num: int
    category: str
    confidence: str          # "high", "medium", "low"
    reason: str
    is_scanned: bool = False


@dataclass
class ClassificationReport:
    pdf_path: str
    categories_used: list[str]
    per_page: list[PageResult]
    metadata: dict = field(default_factory=dict)

    @property
    def category_summary(self) -> dict[str, list[int]]:
        """Returns {category: [page_numbers]}"""
        summary: dict[str, list[int]] = {}
        for p in self.per_page:
            summary.setdefault(p.category, []).append(p.page_num)
        return summary

    @property
    def category_counts(self) -> dict[str, int]:
        return {cat: len(pages) for cat, pages in self.category_summary.items()}

    def to_dict(self) -> dict:
        return {
            "pdf_file": self.pdf_path,
            "total_pages": len(self.per_page),
            "categories_used": self.categories_used,
            "metadata": self.metadata,
            "category_summary": self.category_summary,
            "category_counts": self.category_counts,
            "per_page": [
                {
                    "page": p.page_num,
                    "category": p.category,
                    "confidence": p.confidence,
                    "reason": p.reason,
                }
                for p in self.per_page
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  PDF Classification Report")
        print(f"  File: {self.pdf_path}")
        print(f"  Total Pages: {len(self.per_page)}")
        print(f"{'='*60}")
        print("\n📊 Category Summary:")
        for cat, pages in sorted(self.category_summary.items()):
            print(f"  {cat}: {len(pages)} page(s) → pages {pages}")
        print(f"\n📄 Per-Page Breakdown:")
        for p in self.per_page:
            conf_icon = {"high": "✅", "medium": "⚠️", "low": "❓"}.get(p.confidence, "")
            print(f"  Page {p.page_num:3d}: [{p.category}] {conf_icon}")
            print(f"           Reason: {p.reason}")
        print(f"{'='*60}\n")


def classify_pdf(
    pdf_path: str,
    categories: list[str],
    model: str = "llama3.2",
    use_ocr: bool = False,
    batch_size: int = 5,
    on_progress=None,
) -> ClassificationReport:
    """
    Classify each page of a PDF into one of the user-defined categories.

    Args:
        pdf_path: Path to the PDF
        categories: List of category names (e.g. ["Invoice", "Resume", "Legal"])
        model: Ollama model name
        use_ocr: Enable OCR for scanned PDFs
        batch_size: Pages to classify per LLM call (trade-off: speed vs accuracy)
        on_progress: Optional callback fn(current_page, total_pages) for progress updates
    """
    if not categories:
        raise ValueError("At least one category must be provided.")

    # Always add "Unclassified" so model has an escape hatch
    cats_with_fallback = list(categories)
    if "Unclassified" not in cats_with_fallback:
        cats_with_fallback.append("Unclassified")

    pages_data = extract_text_per_page(pdf_path, use_ocr=use_ocr)
    meta = get_pdf_metadata(pdf_path)
    total = len(pages_data)
    results: list[PageResult] = []

    system_prompt = _build_classifier_system(cats_with_fallback)

    # Process in batches for efficiency
    for batch_start in range(0, total, batch_size):
        batch = pages_data[batch_start: batch_start + batch_size]
        batch_results = _classify_batch(batch, cats_with_fallback, model, system_prompt)
        results.extend(batch_results)

        if on_progress:
            on_progress(min(batch_start + batch_size, total), total)

    return ClassificationReport(
        pdf_path=pdf_path,
        categories_used=cats_with_fallback,
        per_page=results,
        metadata=meta,
    )


def _classify_batch(
    pages: list[dict],
    categories: list[str],
    model: str,
    system_prompt: str,
) -> list[PageResult]:
    """Classify a batch of pages in a single LLM call."""

    pages_text = ""
    for p in pages:
        text_preview = p["text"][:800] if p["text"] else "[No text — possibly scanned image]"
        pages_text += f"\n--- PAGE {p['page_num']} ---\n{text_preview}\n"

    prompt = f"""Classify each of the following PDF pages into one of these categories:
{json.dumps(categories, indent=2)}

For each page, provide:
- page_num: the page number
- category: one of the categories above (use "Unclassified" if none fit)
- confidence: "high", "medium", or "low"
- reason: 1 sentence explaining why

PAGES TO CLASSIFY:
{pages_text}

Return a JSON object with key "pages" containing an array of classification objects."""

    response = ask_json(prompt, model=model, system=system_prompt)

    page_results = []
    classified = response.get("pages", [])

    # Map back to PageResult objects
    classified_by_page = {item.get("page_num"): item for item in classified if isinstance(item, dict)}

    for p in pages:
        pnum = p["page_num"]
        item = classified_by_page.get(pnum, {})
        page_results.append(PageResult(
            page_num=pnum,
            category=item.get("category", "Unclassified"),
            confidence=item.get("confidence", "low"),
            reason=item.get("reason", "No reason provided"),
            is_scanned=p.get("is_scanned", False),
        ))

    return page_results


def _build_classifier_system(categories: list[str]) -> str:
    cats_str = "\n".join(f"- {c}" for c in categories)
    return f"""You are a document page classifier. Your job is to read the text content of PDF pages and assign each page to the most appropriate category.

Available categories:
{cats_str}

Rules:
- Assign EXACTLY ONE category per page
- Use "Unclassified" only when the page genuinely doesn't fit any category
- Base your decision on the actual content and keywords present
- Be consistent: similar content should get the same category
- Respond ONLY with valid JSON, no explanation outside the JSON"""


def load_categories_from_file(path: str) -> list[str]:
    """Load category list from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "categories" in data:
        return data["categories"]
    raise ValueError("Categories file must be a JSON array or {\"categories\": [...]}")
