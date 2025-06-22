import pdfplumber



def extract_text_from_pdf(pdf_path):
    try:    
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    print(f"Sayfa {page_num+1}'de metin bulunamadi.")
        return text
    except Exception as e:
        print(f"Hata olustu: {e}")
        return None