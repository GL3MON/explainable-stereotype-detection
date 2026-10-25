"""Render a saved HTML explanation report to PDF, with every example expanded."""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html")
    parser.add_argument("--pdf")
    args = parser.parse_args(argv)
    source = Path(args.html).resolve()
    output = Path(args.pdf).resolve() if args.pdf else source.with_name("explanations.pdf")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        parser.error("An installed Chrome/Chromium browser is required")
    text = source.read_text().replace("<details>", "<details open>")
    style = """<style>@page{size:A4;margin:14mm}@media print{html,body{font-size:11px;max-width:none;margin:0;padding:0;line-height:1.65}*{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}details{display:block;break-inside:avoid;page-break-inside:avoid;padding:10px;margin:10px 0}summary{display:block;font-weight:bold;font-size:12px}h1{font-size:21px}h3{font-size:12px;margin:8px 0;break-after:avoid}p{margin:6px 0}}</style>"""
    text = text.replace("</style>", "</style>" + style, 1)
    print_html = source.with_name("explanations_print.html")
    print_html.write_text(text)
    with tempfile.TemporaryDirectory(prefix="stereodetect-pdf-") as profile:
        result = subprocess.run(
            [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={output}",
                print_html.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode or not output.exists():
            raise RuntimeError(result.stderr or "PDF rendering failed")
    data = output.read_bytes()
    assert data.startswith(b"%PDF-") and b"%%EOF" in data[-100:]
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    print(f"PDF: {output}; pages={pages}; expanded examples={text.count('<details open>')}")


if __name__ == "__main__":
    main()
