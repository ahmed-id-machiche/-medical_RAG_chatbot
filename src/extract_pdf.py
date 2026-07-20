import os
import re
import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_extracted_text(text: str) -> str:
    # 1. Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Repair hyphenated word breaks at line ends (e.g. "hyper-\ntension" -> "hypertension")
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
    
    # 3. Clean common header/footer noises
    noise_patterns = [
        r'(?i)ministère de la santé et de la protection sociale',
        r'(?i)royaume du maroc',
        r'(?i)guide marocain de\s+\w+',
        r'(?i)plan national de\s+\w+'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)
        
    # 4. Remove isolated page numbers in headers/footers (e.g. "page 4" or just "4" on its own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*[pP]age\s+\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*\d+\s*/\s*\d+\s*\n', '\n', text)
    
    # 5. Normalize multiple whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_all_pdfs():
    print("--- Starting PyMuPDF PDF Text Extraction and Cleaning ---")
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist.")
        return
        
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in data directory.")
        return
        
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        txt_path = os.path.join(CLEAN_DIR, txt_filename)
        
        # Check if already processed
        safe_file = pdf_file.encode('ascii', errors='replace').decode('ascii')
        if os.path.exists(txt_path):
            print(f"Skipping already extracted file: {safe_file}")
            continue
            
        print(f"Extracting {safe_file} using PyMuPDF...")
        try:
            doc = fitz.open(pdf_path)
            full_text_list = []
            
            for page_idx, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    # Format page marker clearly so chunk parser can read it
                    full_text_list.append(f"\n[Page {page_idx + 1}]\n" + page_text)
                    
            raw_text = "".join(full_text_list)
            clean_text = clean_extracted_text(raw_text)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
                
            safe_txt_filename = txt_filename.encode('ascii', errors='replace').decode('ascii')
            print(f"Successfully saved clean text to {safe_txt_filename} ({len(clean_text)} chars)")
        except Exception as e:
            print(f"Error extracting {safe_file}: {str(e)}")

if __name__ == "__main__":
    extract_all_pdfs()
