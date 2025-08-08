import pdfplumber
import os
import re

#PDF and output folder
PDF_PATH = "HR_MANUAL_PATH"
OUTPUT_FOLDER = "hr_docs_by_section"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Read full text from PDF
with pdfplumber.open(PDF_PATH) as pdf:
    full_text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])

#Define section header pattern
section_pattern = re.compile(r"\n(\d\.\d+ [^\n]+)")

#Find all matches and their start positions
matches = list(section_pattern.finditer(full_text))

print(f"Found {len(matches)} sections...")

#Loop through all matches and split text
for i, match in enumerate(matches):
    section_title = match.group(1).strip()
    start_index = match.end()
    end_index = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

    section_text = full_text[start_index:end_index].strip()
    full_section = f"{section_title}\n\n{section_text}"

    #Clean file name
    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", section_title.lower())
    filename = f"section_{safe_title}.txt"
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_section)

    print(f"Saved: {filename}")

print("All sections split and saved.")
