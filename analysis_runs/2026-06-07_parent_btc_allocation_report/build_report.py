# -*- coding: utf-8 -*-
"""Rebuild HTML + PDF + TXT for the parent BTC report from the Markdown source.

Original pipeline used pandoc (+weasyprint), which isn't installed on this box.
Substitute: python-markdown -> standalone HTML (report.css) -> Chrome headless --print-to-pdf.
"""
from __future__ import annotations

import html as htmllib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
STEM = "BTC_parent_allocation_report_2026-07-17"
MD = HERE / f"{STEM}.md"
HTML = HERE / f"{STEM}.html"
PDF = HERE / f"{STEM}.pdf"
TXT = HERE / f"{STEM}.txt"

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="generator" content="python-markdown + chrome-print" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <link rel="stylesheet" href="report.css" />
</head>
<body>
{body}
</body>
</html>
"""


def build_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "sane_lists", "attr_list"],
        output_format="html5",
    )
    return HEAD.format(title=STEM, body=body)


def build_txt(md_text: str) -> str:
    # light plain-text: drop heading/emphasis markers, keep table rows readable
    out = []
    for line in md_text.splitlines():
        s = line
        s = re.sub(r"^#{1,6}\s*", "", s)            # headings
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)    # images
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)  # links -> text
        s = s.replace("**", "").replace("`", "")
        s = s.replace("> ", "")
        out.append(s.rstrip())
    return "\n".join(out).strip() + "\n"


def main() -> None:
    md_text = MD.read_text(encoding="utf-8")

    HTML.write_text(build_html(md_text), encoding="utf-8")
    print("wrote", HTML.name)

    TXT.write_text(build_txt(md_text), encoding="utf-8")
    print("wrote", TXT.name)

    if not Path(CHROME).exists():
        print("WARN: Chrome not found, skipping PDF:", CHROME)
        sys.exit(2)

    with tempfile.TemporaryDirectory() as ud:
        cmd = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={ud}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--print-to-pdf={PDF}",
            HTML.as_uri(),
        ]
        print("running chrome headless...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if PDF.exists():
            print("wrote", PDF.name, f"({PDF.stat().st_size} bytes)")
        else:
            print("PDF FAILED rc=", r.returncode)
            print(r.stderr[-800:])
            sys.exit(1)


if __name__ == "__main__":
    main()
