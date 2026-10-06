import fitz  # PyMuPDF

doc = fitz.open("input.pdf")

text = ""
for page in doc:
    text += page.get_text()

print(text)