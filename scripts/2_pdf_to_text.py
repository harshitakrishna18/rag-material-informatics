from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/pdfs")
TXT_DIR = Path("data/text")
TXT_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            parts.append(f"\n\n--- page {i+1} ---\n{t}")
    return "\n".join(parts)

for pdf in PDF_DIR.glob("*.pdf"):
    txt = pdf_to_text(pdf)
    out = TXT_DIR / (pdf.stem + ".txt")
    out.write_text(txt, encoding="utf-8")
    print("Wrote", out)
