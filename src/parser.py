import os
import re
from pypdf import PdfReader
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, DOCS_DIR

# PDF and text configuration based on PDF instructions
TARGET_MIN_TOKENS = 30
TARGET_MAX_TOKENS = 180
DEFAULT_OVERLAP_TOKENS = 30

THEME_MAPPING = {
    # PDFs (For backward compatibility / reference)
    "PT Diabete (3).pdf": "diabète",
    "GUIDE-M-P.pdf": "nutrition",
    "Guide des Urgences Pédiatriques.pdf": "urgences pédiatriques",
    "Guide des prélévements.pdf": "analyses médicales",
    "Guide_tabac_complet.pdf": "tabagisme",
    "Critères recevabilté PF 2019.pdf": "planification familiale",
    "Plan Stratégique.pdf": "planification stratégique",
    "Politique nationale intégrée de la santé de l'enfant.pdf": "santé de l'enfant",
    "guide_marocain_de_vaccinologie.pdf": "vaccination",
    "Guide National  de prise en charge VF M.pdf": "prise en charge nationale",
    
    # Clean Texts (Extracted via PyMuPDF)
    "PT Diabete (3).txt": "diabète",
    "GUIDE-M-P.txt": "nutrition",
    "guide_nutrition.txt": "nutrition",
    "Guide des Urgences Pédiatriques.txt": "urgences pédiatriques",
    "Guide des prélévements.txt": "analyses médicales",
    "Guide_tabac_complet.txt": "tabagisme",
    "Critères recevabilté PF 2019.txt": "planification familiale",
    "Plan Stratégique.txt": "planification stratégique",
    "Politique nationale intégrée de la santé de l'enfant.txt": "santé de l'enfant",
    "guide_marocain_de_vaccinologie.txt": "vaccination",
    "Guide National  de prise en charge VF M.txt": "prise en charge nationale",
    
    # Markdowns (Fast loading)
    "guide_diabete_maroc.md": "diabète",
    "guide_hypertension_maroc.md": "hypertension",
    "guide_cardio_vasculaire_maroc.md": "maladies cardiovasculaires",
    "guide_maladies_respiratoires_maroc.md": "maladies respiratoires",
    "guide_nutrition_maroc.md": "nutrition",
    "guide_tuberculose_maroc.md": "tuberculose",
    "guide_vaccination_maroc.md": "vaccination",
    "pt_diabete_clean.md": "diabète",
    "guide_tabac_clean.md": "tabagisme"
}


def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in a text.
    In French/English, a word plus its punctuation averages to 1.2 to 1.3 tokens.
    This provides an offline, robust estimation without loading heavy tokenizer weights.
    """
    # Find all words and individual punctuation marks
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return int(len(tokens) * 1.25)

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespaces, removing common PDF extraction
    artifacts, and stripping header/footer noise.
    """
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n', '\n', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.isdigit():
            continue
        if re.match(r'^(page\s+\d+|[Pp]age\s+\d+\/\d+|\d+\s+\/\s+\d+)$', stripped):
            continue
        if re.search(r'Ministère de la Santé et de la Protection Sociale', stripped, re.IGNORECASE):
            continue
        if re.search(r'Guide Marocain de Nutrition', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
        
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using pypdf and applies cleaning.
    """
    if not os.path.exists(pdf_path):
        safe_path = pdf_path.encode('ascii', errors='replace').decode('ascii')
        print(f"PDF file not found: {safe_path}")
        return ""
        
    try:
        reader = PdfReader(pdf_path)
        full_text = []
        safe_name = os.path.basename(pdf_path).encode('ascii', errors='replace').decode('ascii')
        print(f"Extracting text from PDF: {safe_name} ({len(reader.pages)} pages)...")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                full_text.append(f"\n[Page {i+1}]\n" + page_text)
        
        raw_text = "".join(full_text)
        return clean_text(raw_text)
    except Exception as e:
        safe_path = pdf_path.encode('ascii', errors='replace').decode('ascii')
        safe_err = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"Error reading PDF {safe_path}: {safe_err}")
        return ""

def extract_markdown_text(md_path: str) -> str:
    """
    Extracts and cleans text from a markdown file.
    """
    if not os.path.exists(md_path):
        return ""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        return clean_text(content)
    except Exception as e:
        print(f"Error reading Markdown {md_path}: {str(e)}")
        return ""

def chunk_text_by_tokens(text: str, source_name: str, min_tokens: int = TARGET_MIN_TOKENS, max_tokens: int = TARGET_MAX_TOKENS, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list:
    """
    Splits text into chunks of 300 to 800 tokens, with an overlap.
    Maintains sentence boundaries to ensure semantic coherence.
    """
    # Regex split to get sentences while keeping punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_idx = 0
    
    # Store page markers
    page_matches = list(re.finditer(r'\[Page (\d+)\]', text))
    
    def get_page_number(char_index):
        current_page = "1"
        for m in page_matches:
            if m.start() <= char_index:
                current_page = m.group(1)
            else:
                break
        return current_page

    sentence_data = []
    char_pos = 0
    
    # Pre-analyze sentences: calculate character offsets and estimated tokens
    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence_len = len(sentence)
        sentence_tokens = estimate_tokens(sentence)
        sentence_data.append({
            "text": sentence,
            "tokens": sentence_tokens,
            "char_start": char_pos
        })
        # Try to find sentence starting position in the full text
        match_pos = text.find(sentence, char_pos)
        if match_pos != -1:
            char_pos = match_pos + sentence_len
        else:
            char_pos += sentence_len

    i = 0
    while i < len(sentence_data):
        # Accumulate sentences
        current_chunk = []
        current_tokens = 0
        char_start = sentence_data[i]["char_start"]
        
        j = i
        # Force-include the first sentence if it exceeds max_tokens on its own
        if j < len(sentence_data) and sentence_data[j]["tokens"] > max_tokens:
            current_chunk.append(sentence_data[j]["text"])
            current_tokens += sentence_data[j]["tokens"]
            j += 1
        else:
            # Otherwise accumulate sentences normally
            while j < len(sentence_data) and current_tokens + sentence_data[j]["tokens"] <= max_tokens:
                current_chunk.append(sentence_data[j]["text"])
                current_tokens += sentence_data[j]["tokens"]
                j += 1

            
        # If the chunk is smaller than min_tokens but there are more sentences, we still form the chunk
        # or expand it if it's the last few sentences
        
        chunk_content = " ".join(current_chunk).strip()
        clean_chunk_content = re.sub(r'\[Page \d+\]', '', chunk_content).strip()
        
        if clean_chunk_content and current_tokens >= min_tokens or (j == len(sentence_data) and clean_chunk_content):
            page_num = get_page_number(char_start)
            theme = THEME_MAPPING.get(source_name, "santé générale")
            chunks.append({
                "chunk_id": f"{source_name}_chunk_{chunk_idx}",
                "text": clean_chunk_content,
                "source": source_name,
                "page": page_num,
                "tokens": current_tokens,
                "theme": theme
            })
            chunk_idx += 1
            
        # Step back to implement overlap
        # Find how many sentences to step back to achieve the overlap token size
        overlap_accumulated = 0
        step_back_count = 0
        k = j - 1
        while k >= i and overlap_accumulated + sentence_data[k]["tokens"] <= overlap_tokens:
            overlap_accumulated += sentence_data[k]["tokens"]
            step_back_count += 1
            k -= 1
            
        # Set next starting index
        if j == len(sentence_data):
            break # Finished all text
            
        # Ensure we always move forward by at least one sentence to prevent infinite loops
        if step_back_count >= (j - i):
            step_back_count = (j - i) - 1
            
        if step_back_count > 0:
            i = j - step_back_count
        else:
            i = j
            
    return chunks

def is_file_in_scope(filename: str) -> bool:
    return filename.endswith(".md") or filename.endswith(".txt")

def load_and_chunk_all_documents() -> list:
    """
    Scans data_clean folder for PyMuPDF clean texts, and ALSO includes
    documents/ markdown files.
    """
    all_chunks = []
    clean_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_clean")
    
    # 1. Process PyMuPDF cleaned text files in data_clean/
    if os.path.exists(clean_dir):
        print(f"--- Loading and Chunking Clean PDF Texts from {clean_dir} ---")
        for file in os.listdir(clean_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(clean_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    chunks = chunk_text_by_tokens(text, source_name=file)
                    all_chunks.extend(chunks)
                    safe_file = file.encode('ascii', errors='replace').decode('ascii')
                    print(f"Processed cleaned text {safe_file}: generated {len(chunks)} chunks.")
                except Exception as e:
                    safe_file = file.encode('ascii', errors='replace').decode('ascii')
                    print(f"Error processing clean text {safe_file}: {str(e)}")
                    
    # 2. Process markdown files in documents/
    if os.path.exists(DOCS_DIR):
        print(f"--- Loading and Chunking Documents Markdown from {DOCS_DIR} ---")
        for file in os.listdir(DOCS_DIR):
            if file.endswith(".md"):
                if not is_file_in_scope(file):
                    continue
                file_path = os.path.join(DOCS_DIR, file)
                try:
                    text = extract_markdown_text(file_path)
                    chunks = chunk_text_by_tokens(text, source_name=file)
                    all_chunks.extend(chunks)
                    safe_file = file.encode('ascii', errors='replace').decode('ascii')
                    print(f"Processed markdown guide {safe_file}: generated {len(chunks)} chunks.")
                except Exception as e:
                    safe_file = file.encode('ascii', errors='replace').decode('ascii')
                    print(f"Error processing markdown {safe_file}: {str(e)}")
                    
    print(f"Total chunks generated across all documents: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk_all_documents()
    if chunks:
        print("Sample Chunk 0:")
        print(chunks[0])
