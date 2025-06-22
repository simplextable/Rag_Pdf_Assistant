#in this file we are testing a pdf example. we are testing whether pdf_processing is working or not. 

from pdf_processing import extract_text_from_pdf

pdf_path = "test_pdf.pdf"

extracted_text = extract_text_from_pdf(pdf_path)

print("Extracted Text:")
print(extracted_text)