import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "documents")
INDEX_DIR = os.path.join(BASE_DIR, "index")

# Ensure directories exist
for d in [DATA_DIR, DOCS_DIR, INDEX_DIR]:
    os.makedirs(d, exist_ok=True)

# Models
GENERATION_MODEL = "qwen2.5:1.5b"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# RAG configuration
CHUNK_SIZE = 600       # Approximate characters per chunk (approx 150-200 words)
CHUNK_OVERLAP = 100    # Overlap between consecutive chunks

# Official Moroccan Ministry of Health PDFs to scrape/download
OFFICIAL_PDF_URLS = {
    "guide_nutrition.pdf": "https://www.sante.gov.ma/Publications/Guides-Manuels/Documents/GUIDE-M-P.pdf",
    "hta_adulte.pdf": "https://www.sante.gov.ma/Publications/Guides-Manuels/Documents/2019/Recommandations%20de%20Bonnes%20Pratiques%20M%C3%A9dicales%20-%20HTA%20de%20l'adulte.pdf",
    "risque_cardiovasculaire.pdf": "https://www.sante.gov.ma/Publications/Guides-Manuels/Documents/Guide%20risque%20cardiovasculaire.pdf",
    "complications_hta.pdf": "https://www.sante.gov.ma/Publications/Guides-Manuels/Documents/La%20pr%C3%A9vention%20des%20complications.pdf",
    "respiratoire_enfant.pdf": "https://www.sante.gov.ma/Publications/Guides-Manuels/Documents/Guide%20national%20de%20prise%20en%20charge%20des%20principales%20affections%20respiratoires%20de%20l'enfant.pdf"
}

# Evaluation queries for RAG benchmark (combination of French and Darija)
EVALUATION_QUERIES = [
    {
        "query": "Quels sont les trois groupes d'aliments définis dans le guide de nutrition ?",
        "language": "French",
        "expected_topics": ["aliments", "groupes", "nutrition"]
    },
    {
        "query": "ما هي مجموعات الأغذية المذكورة في دليل التغذية؟", # Darija query
        "language": "Darija",
        "expected_topics": ["الأغذية", "التغذية", "مجموعات"]
    },
    {
        "query": "Quelles sont les recommandations nutritionnelles pour une femme enceinte au Maroc ?",
        "language": "French",
        "expected_topics": ["femme enceinte", "grossesse", "nutrition", "fer", "vitamines"]
    },
    {
        "query": "شنو هي النصائح ديال الماكلة للمرا الحاملة ؟", # Darija query
        "language": "Darija",
        "expected_topics": ["الحاملة", "الماكلة", "نصائح"]
    },
    {
        "query": "Comment prévenir l'hypertension artérielle selon les recommandations marocaines ?",
        "language": "French",
        "expected_topics": ["hypertension", "prévenir", "sel", "activité physique"]
    },
    {
        "query": "كيفاش نقدر نحمي راسي من طانسيون ؟", # Darija query
        "language": "Darija",
        "expected_topics": ["طانسيون", "نحمي", "الوقاية"]
    }
]
