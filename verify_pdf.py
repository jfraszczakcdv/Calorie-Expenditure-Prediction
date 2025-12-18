from pypdf import PdfReader
import sys

# Set output encoding to utf-8 just in case
sys.stdout.reconfigure(encoding='utf-8')

try:
    reader = PdfReader("zadanie_AI.pdf")
    content = ""
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"--- Page {i+1} ---\n")
            f.write(text + "\n")
            content += text + "\n"
    print("PDF content saved to pdf_content.txt")
except Exception as e:
    print(f"Error: {e}")
