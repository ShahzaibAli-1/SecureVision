import subprocess
import sys

# Try to install PyMuPDF if not available
try:
    import fitz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF", "-q"])
    import fitz

doc = fitz.open(r'd:\Semester 7\Info Secuirity\Assignment3\ResearchPaper.pdf')
text = ''
for page in doc:
    text += page.get_text()

print(f"Total pages: {doc.page_count}")
print("="*80)
print(text)
