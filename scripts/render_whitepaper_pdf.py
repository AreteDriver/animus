#!/usr/bin/env python3
"""Render a Markdown whitepaper to a styled PDF.

Markdown -> HTML (tables/fenced-code/toc) -> WeasyPrint -> PDF.

Usage:
    python scripts/render_whitepaper_pdf.py \
        docs/whitepapers/ANIMUS_WHITEPAPER_2026-06.md \
        docs/whitepapers/ANIMUS_WHITEPAPER_2026-06.pdf

Requires: markdown (pip), weasyprint (CLI on PATH).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from markdown.extensions.toc import TocExtension


def github_slugify(value: str, separator: str = "-") -> str:
    """Slugify the way GitHub (and the assembled TOC) does it.

    Each whitespace char becomes a separator with NO run-collapsing, so a
    heading like "4.4 Security & Hardening" -> "44-security--hardening"
    (the removed "&" leaves two spaces -> two hyphens). python-markdown's
    default toc slugify collapses those to a single hyphen, which is what
    broke the internal TOC links in the rendered PDF.
    """
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)  # drop punctuation, keep word/space/-
    return re.sub(r"\s", separator, value)  # each space -> separator, no collapse


CSS = """
@page { size: A4; margin: 2cm 2cm 2.2cm 2cm;
  @bottom-center { content: counter(page) " / " counter(pages);
    font-size: 9px; color: #888; } }
body { font-family: "DejaVu Sans", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5px; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22px; border-bottom: 2px solid #222; padding-bottom: 4px;
  margin-top: 1.2em; }
h2 { font-size: 15px; border-bottom: 1px solid #ccc; padding-bottom: 2px;
  margin-top: 1.4em; }
h3 { font-size: 12.5px; color: #333; margin-top: 1.1em; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9px;
  background: #f3f3f3; padding: 1px 3px; border-radius: 3px; }
pre { background: #f6f8fa; padding: 8px; border-radius: 5px; overflow-x: auto;
  font-size: 8.5px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0;
  font-size: 8.8px; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left;
  vertical-align: top; }
th { background: #f0f0f0; }
blockquote { border-left: 3px solid #888; margin: 0.8em 0; padding: 2px 12px;
  color: #444; background: #fafafa; }
a { color: #1a4c8b; text-decoration: none; }
"""


def render(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    # Strip the YAML frontmatter so it does not render as a paragraph.
    text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    html_body = markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            TocExtension(slugify=github_slugify),
            "sane_lists",
        ],
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        tmp_html = f.name
    try:
        subprocess.run(
            ["weasyprint", tmp_html, str(pdf_path)], check=True
        )
    finally:
        Path(tmp_html).unlink(missing_ok=True)
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    render(Path(sys.argv[1]), Path(sys.argv[2]))
