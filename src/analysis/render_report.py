"""Renders a project Markdown document to PDF.

Used to produce docs/09_PROJECT_STATUS_REPORT.pdf from its Markdown source so the
report can be regenerated after every training run instead of drifting out of date.

Converts Markdown -> styled HTML -> PDF via headless Edge/Chrome, which avoids the
GTK/Cairo native dependencies that WeasyPrint needs on Windows.

    python -m src.analysis.render_report --in docs/09_PROJECT_STATUS_REPORT.md

Unless --out is given explicitly, a native "Save As" dialog asks where to write the
PDF, defaulting to the source's own folder and name -- so a run never silently
overwrites the last export in place without the person at the keyboard seeing it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Calibri, system-ui, sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 19pt; border-bottom: 2px solid #2c5f2d; padding-bottom: 5px;
     color: #2c5f2d; margin-top: 0; }
h2 { font-size: 14pt; color: #2c5f2d; margin-top: 20px;
     border-bottom: 1px solid #d0d7d0; padding-bottom: 3px; }
h3 { font-size: 11.5pt; color: #3a3a3a; margin-top: 15px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt; }
th { background: #eef3ee; text-align: left; }
th, td { border: 1px solid #c5cec5; padding: 5px 7px; vertical-align: top; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, monospace; font-size: 9pt; }
pre { background: #f7f7f7; border: 1px solid #ddd; border-left: 3px solid #2c5f2d;
      padding: 8px 10px; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #b8860b; background: #fdf9ef; margin: 10px 0;
             padding: 7px 12px; color: #4a4a4a; }
strong { color: #1a1a1a; }
li { margin: 2px 0; }
h2, h3 { page-break-after: avoid; }
table, pre, blockquote { page-break-inside: avoid; }
"""


def ask_save_path(default_path: str) -> str | None:
    """Native "Save As" dialog defaulting to `default_path`. Returns None if the
    person cancelled, or if no display is available to show a dialog at all (e.g.
    a headless CI run) -- callers must not treat None as "use the default silently",
    since that would defeat the point of asking.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        print("No tkinter available to show a Save As dialog.", file=sys.stderr)
        return None

    default_dir = os.path.dirname(os.path.abspath(default_path)) or "."
    default_name = os.path.basename(default_path)

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.asksaveasfilename(
            title="Save project report as",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return chosen or None


def find_browser() -> str | None:
    candidates = [
        r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        r"C:/Program Files/Google/Chrome/Application/chrome.exe",
        r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for name in ("msedge", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return None


def render(md_path: str, pdf_path: str) -> None:
    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "codehilite", "sane_lists", "toc"]
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )

    browser = find_browser()
    if browser is None:
        raise SystemExit("No Edge/Chrome found to print the PDF.")

    tmp_dir = tempfile.mkdtemp(prefix="report_render_")
    tmp_html = os.path.join(tmp_dir, "report.html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html)

    pdf_path = os.path.abspath(pdf_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file:///{tmp_html.replace(os.sep, '/')}",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    # Headless Chromium writes the file asynchronously on exit.
    for _ in range(40):
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            break
        time.sleep(0.25)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not os.path.exists(pdf_path):
        raise SystemExit(f"Browser did not produce {pdf_path}")
    print(f"Wrote {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default="docs/09_PROJECT_STATUS_REPORT.md")
    parser.add_argument(
        "--out", dest="dst", default=None,
        help="Write here without prompting. Omit to choose the destination in a "
             "Save As dialog instead.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"Missing input: {args.src}")

    default_dst = os.path.splitext(args.src)[0] + ".pdf"
    if args.dst:
        dst = args.dst
    else:
        dst = ask_save_path(default_dst)
        if dst is None:
            sys.exit("Save cancelled; no PDF written.")

    render(args.src, dst)


if __name__ == "__main__":
    main()
