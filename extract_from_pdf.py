import pdfplumber
import os

#Path to your PDF
PDF_PATH = "HR_MANUAL_PATH"

#Output file
OUTPUT_PATH = os.path.join("hr_docs", "staff_manual.txt")

#Extract full text
with pdfplumber.open(PDF_PATH) as pdf:
    full_text = ""
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text:
            full_text += f"\n\n--- Page {page_num} ---\n\n{text}"

# Save to .txt
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted text saved to: {OUTPUT_PATH}")
