"""
ingest.py
Reads all raw source files from data/raw/, extracts clean text from each
(PDF / HTML / TXT), and writes one .txt file per source into data/processed/
along with a manifest.json that records source filename -> original URL.

Usage:
    python src/ingest.py
"""

import os
import json
import pdfplumber
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Fill this in with the original URL (or "local upload") for each raw filename.
# This is used only for the "source display" feature later, so answers can
# point back to where the information came from.
SOURCE_URLS = {
    "campusCodeOfConduct.pdf": "https://gymkhana.iitb.ac.in/students/docs/campusCodeOfConduct.pdf",
    "housing.html": "https://gymkhana.iitb.ac.in/students/housing.html",
    "Hostel10_Rules.pdf": "https://gymkhana.iitb.ac.in/~hostel10/documents/Hostel10_Rules.pdf",
    "mess.html": "https://gymkhana.iitb.ac.in/~hostel10/mess.html",
    "Menu-week1.pdf": "https://gymkhana.iitb.ac.in/~hostel16/messMenus/Menu-week1.pdf",
    "Infrastructure_Details.pdf": "https://notopedia-uploads.s3.us-east-2.amazonaws.com/college_brochure/IIT%20Bombay/Infrastructure%20Details.pdf",
    "Handbook_HSS.pdf": "Local upload (IITB HSS Department Fresher's Handbook)",
    # .txt fallback versions (pre-extracted, included so the pipeline is runnable
    # end-to-end immediately; replace data/raw/ with your own downloaded PDFs/HTML
    # for the "real" ingestion the assignment asks for)
    "campusCodeOfConduct.txt": "https://gymkhana.iitb.ac.in/students/docs/campusCodeOfConduct.pdf",
    "housing.txt": "https://gymkhana.iitb.ac.in/students/housing.html",
    "Hostel10_Rules.txt": "https://gymkhana.iitb.ac.in/~hostel10/documents/Hostel10_Rules.pdf",
    "mess.txt": "https://gymkhana.iitb.ac.in/~hostel10/mess.html",
    "Menu-week1.txt": "https://gymkhana.iitb.ac.in/~hostel16/messMenus/Menu-week1.pdf",
    "Infrastructure_Details.txt": "https://notopedia-uploads.s3.us-east-2.amazonaws.com/college_brochure/IIT%20Bombay/Infrastructure%20Details.pdf",
    "Handbook_HSS.txt": "Local upload (IITB HSS Department Fresher's Handbook)",
    "housing.pdf": "https://gymkhana.iitb.ac.in/students/housing.html",
    "mess.pdf": "https://gymkhana.iitb.ac.in/~hostel10/mess.html",
}


def extract_pdf_text(filepath: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_html_text(filepath: str) -> str:
    """Extract visible text from a saved HTML page using BeautifulSoup."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Remove script/style tags before extracting text
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_txt_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def process_all():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    manifest = {}

    if not os.path.isdir(RAW_DIR):
        print(f"[!] Raw data folder not found: {RAW_DIR}")
        print("    Create it and place your downloaded PDFs/HTML files there.")
        return

    files = sorted(os.listdir(RAW_DIR))
    if not files:
        print(f"[!] No files found in {RAW_DIR}. Add your downloaded source files first.")
        return

    for filename in files:
        filepath = os.path.join(RAW_DIR, filename)
        ext = filename.lower().split(".")[-1]

        try:
            if ext == "pdf":
                text = extract_pdf_text(filepath)
            elif ext in ("html", "htm"):
                text = extract_html_text(filepath)
            elif ext == "txt":
                text = extract_txt_text(filepath)
            else:
                print(f"[skip] Unsupported file type: {filename}")
                continue
        except Exception as e:
            print(f"[error] Failed to process {filename}: {e}")
            continue

        if not text.strip():
            print(f"[warn] No extractable text in {filename} (might be a scanned/image PDF).")
            continue

        out_name = os.path.splitext(filename)[0] + ".txt"
        out_path = os.path.join(PROCESSED_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        manifest[out_name] = {
            "original_file": filename,
            "source_url": SOURCE_URLS.get(filename, "URL not recorded — add it to SOURCE_URLS in ingest.py"),
        }
        print(f"[ok] Processed {filename} -> {out_name} ({len(text)} chars)")

    manifest_path = os.path.join(PROCESSED_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest)} documents processed. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    process_all()
