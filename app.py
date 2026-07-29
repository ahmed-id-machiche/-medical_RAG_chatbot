# -*- coding: utf-8 -*-
import os
import dotenv
dotenv.load_dotenv()

import sys
import threading
import time
import importlib
import json
import uuid
import base64
from io import BytesIO

import src.config
import src.vector_store
import src.rag_pipeline
import src.parser
import src.downloader
import src.extract_pdf

importlib.reload(src.config)
importlib.reload(src.vector_store)
importlib.reload(src.rag_pipeline)
importlib.reload(src.parser)
importlib.reload(src.downloader)
importlib.reload(src.extract_pdf)

import streamlit as st
from src.config import INDEX_DIR, DATA_DIR, DOCS_DIR
from src.rag_pipeline import RAGPipeline
from src.parser import load_and_chunk_all_documents
from src.downloader import download_official_pdfs, create_fallback_documents
from src.extract_pdf import extract_all_pdfs

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Helper to load images in Base64 for CSS injection
def get_base64_image(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    return ""

# 1. STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="Tbibk",
    page_icon="🩺",
    layout="centered"
)

# Initialize global shared RAG pipeline once and cache in resource memory
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

pipeline = get_rag_pipeline()

# Background Indexing Status state
if "is_indexing" not in st.session_state:
    st.session_state.is_indexing = False

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize conversation ID
if "conv_id" not in st.session_state:
    st.session_state.conv_id = ""

# Initialize clinical inputs in session state to persist them across pages
if "poids" not in st.session_state:
    st.session_state.poids = 70.0
if "taille" not in st.session_state:
    st.session_state.taille = 170.0
if "age" not in st.session_state:
    st.session_state.age = 45
if "sexe" not in st.session_state:
    st.session_state.sexe = "Femme"
if "pas" not in st.session_state:
    st.session_state.pas = 120
if "tabac" not in st.session_state:
    st.session_state.tabac = "Non"

def check_and_build_index():
    if pipeline.vector_store.is_empty():
        st.session_state.is_indexing = True
        try:
            print("--- Starting Background Indexing for Tbibk (Streamlit) ---")
            download_official_pdfs()
            create_fallback_documents()
            extract_all_pdfs()
            chunks = load_and_chunk_all_documents()
            pipeline.vector_store.add_chunks(chunks)
            pipeline.vector_store.save()
            print("--- Background Indexing Completed Successfully! ---")
        except Exception as e:
            print(f"Error during background indexing: {str(e)}")
        finally:
            st.session_state.is_indexing = False

# Trigger background indexing once per session
if "indexing_started" not in st.session_state:
    st.session_state.indexing_started = True
    threading.Thread(target=check_and_build_index, daemon=True).start()

# ================= PERSISTENCE: CONVERSATION HISTORY =================
CONVS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations")
os.makedirs(CONVS_DIR, exist_ok=True)

def save_current_conversation():
    if "conv_id" not in st.session_state or not st.session_state.conv_id:
        st.session_state.conv_id = str(uuid.uuid4())
        
    if not st.session_state.messages:
        return
        
    # Extract title from the first message
    first_msg = st.session_state.messages[0]["content"]
    title = first_msg[:24] + "..." if len(first_msg) > 24 else first_msg
    
    data = {
        "id": st.session_state.conv_id,
        "title": title,
        "messages": st.session_state.messages,
        "poids": st.session_state.poids,
        "taille": st.session_state.taille,
        "age": st.session_state.age,
        "sexe": st.session_state.sexe,
        "pas": st.session_state.pas,
        "tabac": st.session_state.tabac
    }
    
    filepath = os.path.join(CONVS_DIR, f"{st.session_state.conv_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_conversations():
    convs = []
    if os.path.exists(CONVS_DIR):
        for fname in os.listdir(CONVS_DIR):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(CONVS_DIR, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        convs.append({
                            "id": data["id"],
                            "title": data["title"],
                            "timestamp": os.path.getmtime(os.path.join(CONVS_DIR, fname))
                        })
                except Exception:
                    pass
    # Sort by timestamp descending
    convs.sort(key=lambda x: x["timestamp"], reverse=True)
    return convs

def load_conversation(conv_id):
    filepath = os.path.join(CONVS_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.conv_id = data["id"]
                st.session_state.messages = data["messages"]
                st.session_state.poids = data.get("poids", 70.0)
                st.session_state.taille = data.get("taille", 170.0)
                st.session_state.age = data.get("age", 45)
                st.session_state.sexe = data.get("sexe", "Femme")
                st.session_state.pas = data.get("pas", 120)
                st.session_state.tabac = data.get("tabac", "Non")
        except Exception:
            pass

# Helper to reshape Arabic text and handle BiDi right-to-left layout for ReportLab PDF
def ar(text: str) -> str:
    if not text:
        return ""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

# Register Tahoma font for native Arabic rendering in ReportLab PDF
try:
    if os.path.exists("C:/Windows/Fonts/tahoma.ttf"):
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont('Tahoma', 'C:/Windows/Fonts/tahoma.ttf'))
        pdfmetrics.registerFont(TTFont('Tahoma-Bold', 'C:/Windows/Fonts/tahomabd.ttf'))
        FONT_AR = 'Tahoma'
        FONT_AR_BOLD = 'Tahoma-Bold'
    else:
        FONT_AR = 'Helvetica'
        FONT_AR_BOLD = 'Helvetica-Bold'
except Exception:
    FONT_AR = 'Helvetica'
    FONT_AR_BOLD = 'Helvetica-Bold'

def generate_patient_pdf(imc, imc_status, risk_status, messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    primary_color = colors.HexColor("#1E3A8A")
    secondary_color = colors.HexColor("#2563EB")
    text_color = colors.HexColor("#1E293B")
    
    fr_header_style = ParagraphStyle(
        name='FrHeader',
        fontName=FONT_AR_BOLD,
        fontSize=8.5,
        leading=12,
        textColor=primary_color,
        alignment=0
    )
    
    ar_header_style = ParagraphStyle(
        name='ArHeader',
        fontName=FONT_AR_BOLD,
        fontSize=8.5,
        leading=12,
        textColor=primary_color,
        alignment=2
    )
    
    doc_title_style = ParagraphStyle(
        name='DocTitle',
        fontName=FONT_AR_BOLD,
        fontSize=12.5,
        leading=16,
        textColor=primary_color,
        alignment=1,
        spaceBefore=10,
        spaceAfter=12
    )
    
    table_header_style = ParagraphStyle(
        name='TableHeader',
        fontName=FONT_AR_BOLD,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0F172A")
    )
    
    table_body_style = ParagraphStyle(
        name='TableBody',
        fontName=FONT_AR,
        fontSize=9.5,
        leading=13,
        textColor=text_color
    )
    
    disclaimer_style = ParagraphStyle(
        name='Disclaimer',
        fontName=FONT_AR,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        alignment=1
    )

    story = []
    
    # 1. Dual-Language Official Moroccan Header (French Left, Logo Center, Reshaped Arabic Right)
    left_header = Paragraph(
        "<b>ROYAUME DU MAROC</b><br/>"
        "<font color='#475569' size=7.5>Ministère de la Santé</font><br/>"
        "<b><font color='#2563EB' size=8>TBIBK — Assistant RAG</font></b>",
        fr_header_style
    )
    
    right_header = Paragraph(
        f"<b>{ar('المملكة المغربية')}</b><br/>"
        f"<font color='#475569' size=7.5>{ar('وزارة الصحة والحماية الاجتماعية')}</font><br/>"
        f"<b><font color='#2563EB' size=8>{ar('طبيبك — المساعد الطبي')}</font></b>",
        ar_header_style
    )
    
    logo_path = os.path.join(os.path.dirname(__file__), "tbibk_logo.png")
    if os.path.exists(logo_path):
        logo_img = Image(logo_path, width=48, height=48)
    else:
        logo_img = Paragraph("<b>TBIBK</b>", fr_header_style)
        
    header_table = Table([[left_header, logo_img, right_header]], colWidths=[200, 100, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(header_table)
    
    # Divider line under header
    divider_table = Table([[""]], colWidths=[500], rowHeights=[2])
    divider_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,-1), 1.5, primary_color),
    ]))
    story.append(divider_table)
    story.append(Spacer(1, 8))
    
    # Document Title
    story.append(Paragraph(f"<b>FICHE CLINIQUE DE CONSULTATION | {ar('بطاقة الاستشارة الطبية')}</b>", doc_title_style))
    story.append(Spacer(1, 8))
    
    # 2. Bilingual Clinical Parameters Table (Reshaped Arabic)
    param_data = [
        [
            Paragraph(f"<b>Indicateur Évalué ({ar('المؤشر السريري')})</b>", table_header_style),
            Paragraph(f"<b>Résultat & Statut ({ar('النتيجة والتقييم')})</b>", table_header_style)
        ],
        [
            Paragraph(f"Poids / Taille ({ar('الوزن والطول')})", table_body_style),
            Paragraph(f"{st.session_state.poids} kg / {st.session_state.taille} cm", table_body_style)
        ],
        [
            Paragraph(f"Indice de Masse Corporelle (IMC / {ar('مؤشر كتلة الجسم')})", table_body_style),
            Paragraph(f"{imc:.1f} ({imc_status})", table_body_style)
        ],
        [
            Paragraph(f"Risque Cardiovasculaire (HTA / {ar('خطر القلب والضغط')})", table_body_style),
            Paragraph(f"{risk_status}", table_body_style)
        ]
    ]
    
    t = Table(param_data, colWidths=[270, 230])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # 3. Disclaimer Footer
    story.append(Paragraph(
        f"<i>Avertissement : Ce document est une fiche d'information automatique et ne remplace pas une consultation médicale.<br/>"
        f"{ar('تنبيه: هذه البطاقة معلوماتية تم إنشاؤها تلقائياً ولا تعوض الاستشارة الطبية الفعلية.')}</i>",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Load 3D icons as Base64 strings
chat_icon_b64 = get_base64_image("3d_chat_icon.png")
imc_icon_b64 = get_base64_image("3d_imc_icon.png")
heart_icon_b64 = get_base64_image("3d_heart_icon.png")
report_icon_b64 = get_base64_image("3d_report_icon.png")

# ================= SIDEBAR: NAVIGATION MENU (DRAWER STYLE) =================
# Injecting CSS directly inside st.sidebar.markdown to ensure it overrides sidebar element classes
st.sidebar.markdown(f"""
<style>
    /* Hide the radio button circle indicators aggressively without hiding the text */
    div[role="radiogroup"] label input[type="radio"],
    div[role="radiogroup"] label svg,
    div[role="radiogroup"] label [role="presentation"] {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
    }}
    
    /* Hide all circle dot wrappers and spans directly following the radio input, leaving only the text */
    div[role="radiogroup"] label input[type="radio"] ~ div:not(:last-child) {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }}
    div[role="radiogroup"] label input[type="radio"] ~ span {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }}
    
    /* Force show the text container (which is the last child sibling of the input) */
    div[role="radiogroup"] label input[type="radio"] ~ div:last-child {{
        display: flex !important;
        opacity: 1 !important;
    }}
    
    /* Ensure markdown and paragraph text elements are fully visible and readable */
    div[role="radiogroup"] label [data-testid="stMarkdownContainer"],
    div[role="radiogroup"] label p {{
        display: block !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    /* Style the list labels as clean flat menu items */
    div[role="radiogroup"] label {{
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
    }}
    
    /* Change the writing style / typography of the section names */
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span,
    div[role="radiogroup"] label [data-testid="stMarkdownContainer"] {{
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important; /* White text on blue sidebar background */
        opacity: 0.90 !important;
        letter-spacing: -0.015em !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    /* Inject 3D Icons as ::before pseudo-elements */
    div[role="radiogroup"] label:nth-of-type(1)::before {{
        content: "" !important;
        display: inline-block !important;
        width: 28px !important;
        height: 28px !important;
        margin-right: 12px !important;
        background-image: url("data:image/png;base64,{chat_icon_b64}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }}
    div[role="radiogroup"] label:nth-of-type(2)::before {{
        content: "" !important;
        display: inline-block !important;
        width: 28px !important;
        height: 28px !important;
        margin-right: 12px !important;
        background-image: url("data:image/png;base64,{imc_icon_b64}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }}
    div[role="radiogroup"] label:nth-of-type(3)::before {{
        content: "" !important;
        display: inline-block !important;
        width: 28px !important;
        height: 28px !important;
        margin-right: 12px !important;
        background-image: url("data:image/png;base64,{heart_icon_b64}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }}
    div[role="radiogroup"] label:nth-of-type(4)::before {{
        content: "" !important;
        display: inline-block !important;
        width: 28px !important;
        height: 28px !important;
        margin-right: 12px !important;
        background-image: url("data:image/png;base64,{report_icon_b64}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }}
    
    /* Hover effect */
    div[role="radiogroup"] label:hover {{
        background-color: rgba(255, 255, 255, 0.15) !important;
    }}
    div[role="radiogroup"] label:hover p,
    div[role="radiogroup"] label:hover span,
    div[role="radiogroup"] label:hover [data-testid="stMarkdownContainer"] {{
        color: #FFFFFF !important;
        opacity: 1 !important;
    }}
    
    /* Style the checked option - Solid flat White box with Blue text! */
    div[role="radiogroup"] label:has(input:checked) {{
        background-color: #FFFFFF !important;
    }}
    
    div[role="radiogroup"] label:has(input:checked) p,
    div[role="radiogroup"] label:has(input:checked) span,
    div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] {{
        color: #549FC4 !important; /* Blue text on active white box */
        font-weight: 700 !important;
        opacity: 1 !important;
    }}
    
    /* Sidebar buttons styling */
    [data-testid="stSidebar"] button {{
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        opacity: 0.90 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 15.5px !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        transition: all 0.15s ease !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] button:hover {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
    }}
    
    /* New Chat button style (first button) */
    [data-testid="stSidebar"] [data-testid="element-container"]:nth-of-type(2) button {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px dashed rgba(255, 255, 255, 0.6) !important;
        font-weight: 600 !important;
        font-size: 16.5px !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
        margin-bottom: 15px !important;
    }}
    [data-testid="stSidebar"] [data-testid="element-container"]:nth-of-type(2) button:hover {{
        border-color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.25) !important;
        color: #FFFFFF !important;
    }}
</style>
""", unsafe_allow_html=True)

# Render logo at the top of the sidebar
if os.path.exists("tbibk_logo.png"):
    st.sidebar.image("tbibk_logo.png", use_container_width=True)
else:
    st.sidebar.markdown("""
    <div style="padding-top: 15px; margin-bottom: 10px; padding-left: 10px;">
        <span style="font-size: 20px; font-weight: 700; color: #FFFFFF; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em;">🏥 Tbibk Workspace</span>
    </div>
    """, unsafe_allow_html=True)

# ➕ New Chat Button in the sidebar
if st.sidebar.button("➕ Nouvelle Discussion", use_container_width=True):
    st.session_state.conv_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.navigation_page = "Chatbot Tbibk"
    st.rerun()

# Styled Radio Button Navigation (No emojis, styled via CSS pseudo-elements)
page = st.sidebar.radio(
    "Menu :",
    ["Chatbot Tbibk", "Calculateur d'IMC", "Risque Cardiovasculaire", "Fiche Patient"],
    key="navigation_page"
)

# Render past conversations list
st.sidebar.markdown("<hr style='margin: 15px 0; border-color: rgba(255, 255, 255, 0.2);'/>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='font-size: 11px; font-weight: 700; color: #FFFFFF; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em; padding-left: 10px; display: block; margin-bottom: 8px;'>🕒 Discussions passées</span>", unsafe_allow_html=True)

past_convs = list_conversations()
if not past_convs:
    st.sidebar.markdown("<span style='font-size: 12px; color: #FFFFFF; opacity: 0.6; padding-left: 10px; font-style: italic;'>Aucun historique</span>", unsafe_allow_html=True)
else:
    for conv in past_convs:
        if st.sidebar.button(f"💬 {conv['title']}", key=f"conv_{conv['id']}", use_container_width=True):
            load_conversation(conv['id'])
            st.session_state.navigation_page = "Chatbot Tbibk"
            st.rerun()

# 2. INJECT CUSTOM THEMING CSS FOR SINGLE-COLUMN CLEAN LAYOUT
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Force main layout to be centered and clean */
    .block-container {
        max-width: 800px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        margin: 0 auto !important;
    }
    
    /* Set page background to the exact custom blue color on all outer containers and override gradients */
    html,
    body,
    .stApp,
    .main,
    [data-testid="stMain"],
    [data-testid="stMainViewContainer"],
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    div[class*="stBottom"],
    footer {
        background: #549FC4 !important;
        background-image: none !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Make all bottom bar wrappers transparent, except the actual chat input pill itself */
    [data-testid="stBottom"] *:not([data-testid="stChatInput"]):not([data-testid="stChatInput"] *),
    [data-testid="stBottomBlockContainer"] *:not([data-testid="stChatInput"]):not([data-testid="stChatInput"] *),
    div[class*="stBottom"] *:not([data-testid="stChatInput"]):not([data-testid="stChatInput"] *) {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Ensure all Streamlit widget labels are white on the blue background */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    label p {
        color: #FFFFFF !important;
    }
    
    /* Style image containers (logos) to be completely transparent without white backgrounds */
    [data-testid="stImage"] img {
        background-color: transparent !important;
        border-radius: 0px !important;
        padding: 0px !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Hide the default Streamlit fullscreen overlay button on all images completely */
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* Style the sidebar elements in solid brand blue */
    [data-testid="stSidebar"] {
        background-color: #549FC4 !important;
        border-right: 1px solid #488EAF !important;
    }
    
    /* Set Sidebar header text and dividers to white */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }
    
    /* Title typography - Clean white centered */
    .clean-title {
        font-size: 38px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-align: center !important;
        margin-top: 4vh !important;
        margin-bottom: 25px !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.03em !important;
    }
    
    .clean-subtitle {
        font-size: 14.5px !important;
        color: rgba(255, 255, 255, 0.85) !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.5 !important;
    }
    
    /* Clean message containers - Styled as elegant white cards on the blue background */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        border: 1px solid #E2ECF2 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Fix chat message text color to remain dark grey for readability inside the white bubbles */
    .stChatMessage p,
    .stChatMessage span,
    .stChatMessage div {
        color: #1F2937 !important;
    }
    
    /* Language and translation sub-note style */
    .lang-badge {
        font-size: 10px;
        padding: 2px 6px;
        border-radius: 4px;
        color: #4B5563;
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }
    .reformulation-note {
        font-size: 11px;
        color: #6B7280;
        margin-bottom: 6px;
        font-style: italic;
    }
    
    /* Clean minimal result card */
    .premium-card {
        background: #FFFFFF !important;
        border: 1px solid #E2ECF2 !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.2s ease !important;
    }
    .premium-card p,
    .premium-card td,
    .premium-card th {
        color: #1F2937 !important;
    }
    
    /* Automatically align Arabic text to right (RTL) */
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.7;
        font-size: 15.5px;
        color: #111827 !important;
    }
    
    /* Premium ChatGPT-like Chat Input styling */
    div[data-testid="stChatInput"] {
        border: 1px solid #E5E7EB !important;
        border-radius: 26px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important;
        padding: 6px 12px !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #549FC4 !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 15px !important;
        color: #111827 !important;
    }
    
    /* Custom Streamlit Button overrides for suggestion chips - Solid white with blue text */
    div[data-testid="column"] button {
        background-color: #FFFFFF !important;
        color: #549FC4 !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 20px !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    div[data-testid="column"] button:hover {
        background-color: #F0F7FB !important;
        border-color: #549FC4 !important;
        color: #3B82A6 !important;
    }
    
    /* Style all action buttons (like download buttons) with a darker accent blue color */
    div.stDownloadButton button {
        background-color: #3B82A6 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    div.stDownloadButton button:hover {
        background-color: #2D6582 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Style Warning Box (Notification alert) to match the brand blue */
    div[data-testid="stNotification"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
    }
    div[data-testid="stNotification"] p {
        color: #FFFFFF !important;
    }
    
    /* Clean inputs styled to remain white with dark text */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border-color: #E2ECF2 !important;
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="input"] input {
        color: #111827 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] {
        border-radius: 10px !important;
        border-color: #E2ECF2 !important;
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] div {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. PAGE RENDERING BY NAVIGATION PAGE
if page == "Chatbot Tbibk":
    # Centered Logo
    logo_col1, logo_col2, logo_col3 = st.columns([1.5, 1, 1.5])
    with logo_col2:
        if os.path.exists("tbibk_logo.png"):
            st.image("tbibk_logo.png", use_container_width=True)

    # Centered Name TBIBK below the logo
    st.markdown("<div style='text-align: center; margin-top: -5px; margin-bottom: 20px;'><span style='font-size: 34px; font-weight: 800; color: #FFFFFF; font-family: \"Outfit\", sans-serif; letter-spacing: 0.05em;'>TBIBK</span></div>", unsafe_allow_html=True)

    # Empty history header: "Comment puis-je vous aider ?"
    if len(st.session_state.messages) == 0:
        st.markdown("<div style='text-align: center; margin-top: 0px; margin-bottom: 25px;'><span style='font-size: 24px; font-weight: 600; color: rgba(255, 255, 255, 0.9); font-family: \"Inter\", sans-serif;'>Comment puis-je vous aider ?</span></div>", unsafe_allow_html=True)

    # Warning alert box
    st.warning("⚠️ **Avertissement :** Tbibk est un assistant d'information médicale basé sur les sources officielles du Ministère de la Santé du Maroc. Il ne remplace pas l'avis d'un professionnel de santé.")

    # Display Indexing banner if indexing is in progress
    if st.session_state.is_indexing:
        st.info("⚡ **Indexation en cours :** La base de données locale se charge en arrière-plan. Veuillez patienter quelques instants...")

    # Render messages history
    for index, msg in enumerate(st.session_state.messages):
        avatar_icon = "👤" if msg["role"] == "user" else "tbibk_logo.png"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            if msg["role"] == "assistant":
                # Badges
                is_darija = msg.get("is_darija", False)
                if is_darija:
                    st.markdown('<div class="lang-badge">Darija 🇲🇦</div>', unsafe_allow_html=True)
                    if msg.get("query_fr"):
                        st.markdown(f'<div class="reformulation-note">Question traduite : "{msg["query_fr"]}"</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="lang-badge">Français 🇫🇷</div>', unsafe_allow_html=True)

                # Conditional rendering for RTL support
                if is_darija:
                    st.markdown(f'<div class="arabic-text">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    # Input Chat Block
    if prompt := st.chat_input("Posez votre question médicale..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Save conversation to disk
        save_current_conversation()
        st.rerun()

    # Process query
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant", avatar="tbibk_logo.png"):
            with st.spinner("Recherche dans les sources officielles..."):
                try:
                    res = pipeline.answer_query(user_query, top_k=4)
                    
                    # Format sources for backend storage
                    sources_metadata = [
                        {
                            "source": chunk["source"],
                            "page": chunk.get("page", "N/A"),
                            "text": chunk["text"]
                        }
                        for chunk, score in res["retrieved_sources"]
                    ]
                    
                    # Add response to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["response"],
                        "is_darija": res["is_darija"],
                        "query_fr": res["query_fr"],
                        "sources": sources_metadata
                    })
                    # Save conversation to disk
                    save_current_conversation()
                    st.rerun()
                except Exception as e:
                    st.error(f"Désolé, une erreur est survenue lors du traitement de votre requête : {str(e)}")

elif page == "Calculateur d'IMC":
    st.markdown('<div class="clean-title">⚖️ Indice de Masse Corporelle (IMC)</div>', unsafe_allow_html=True)
    st.markdown('<div class="clean-subtitle">Évaluez votre poids par rapport à votre taille selon les directives du Ministère de la Santé du Maroc.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        poids = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=st.session_state.poids, step=1.0)
        st.session_state.poids = poids
        save_current_conversation()
    with col2:
        taille = st.number_input("Taille (cm)", min_value=100.0, max_value=220.0, value=st.session_state.taille, step=1.0)
        st.session_state.taille = taille
        save_current_conversation()
        
    imc = poids / ((taille / 100.0) ** 2)
    
    if imc < 18.5:
        imc_status = "Maigreur"
        imc_color = "#EAB308" # Yellow
        imc_desc = "Poids insuffisant. Veillez à enrichir votre alimentation avec des aliments denses en nutriments."
    elif imc < 25.0:
        imc_status = "Normal"
        imc_color = "#22C55E" # Green
        imc_desc = "Poids idéal. Vous êtes dans la plage de santé recommandée. Continuez ainsi !"
    elif imc < 30.0:
        imc_status = "Surpoids"
        imc_color = "#F97316" # Orange
        imc_desc = "Surpoids léger. Limitez les graisses, les sucres raffinés, et faites au moins 30 minutes d'activité physique par jour."
    else:
        imc_status = "Obésité"
        imc_color = "#EF4444" # Red
        imc_desc = "Obésité. Risque accru de maladies cardiovasculaires, d'hypertension et de diabète. Nous vous conseillons de consulter votre centre de santé."

    st.markdown(f"""
    <div class="premium-card" style="text-align: center;">
        <span style="font-size: 13px; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Votre Résultat d'IMC</span>
        <div style="font-size: 54px; font-weight: 800; color: {imc_color}; margin-top: 12px; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">{imc:.1f}</div>
        <div style="font-size: 20px; font-weight: 700; color: {imc_color}; font-family: 'Outfit', sans-serif;">{imc_status}</div>
        <div style="font-size: 14px; color: #4B5563; margin-top: 20px; border-top: 1px solid #E2ECF2; padding-top: 15px; line-height: 1.5;">💡 <b>Conseil personnalisé :</b> {imc_desc}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    st.info("ℹ️ **À propos de l'IMC :** L'Indice de Masse Corporelle est un indicateur standard approuvé par l'Organisation Mondiale de la Santé (OMS) et le Ministère de la Santé marocain pour dépister les risques liés au poids.")

elif page == "Risque Cardiovasculaire":
    st.markdown('<div class="clean-title">💓 Risque Cardio-Vasculaire & HTA</div>', unsafe_allow_html=True)
    st.markdown('<div class="clean-subtitle">Évaluez votre risque cardiovasculaire et votre tension artérielle selon les normes cliniques officielles du Maroc.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Âge", min_value=18, max_value=100, value=st.session_state.age)
        st.session_state.age = age
        sexe = st.selectbox("Sexe", ["Femme", "Homme"], index=0 if st.session_state.sexe == "Femme" else 1)
        st.session_state.sexe = sexe
        save_current_conversation()
    with col2:
        pas = st.slider("Tension Systolique (PAS en mmHg)", min_value=80, max_value=220, value=st.session_state.pas)
        st.session_state.pas = pas
        tabac = st.selectbox("Fumeur", ["Non", "Oui"], index=0 if st.session_state.tabac == "Non" else 1)
        st.session_state.tabac = tabac
        save_current_conversation()
        
    # Simple algorithm inspired by SCORE HTA guidelines
    risk_score = 0
    if age >= 50: risk_score += 2
    if pas >= 140: risk_score += 3
    if tabac == "Oui": risk_score += 2

    if risk_score <= 1:
        risk_status = "Faible"
        risk_color = "#22C55E"
        risk_desc = "Risque cardiovasculaire faible. Continuez à maintenir une alimentation saine et équilibrée."
    elif risk_score <= 4:
        risk_status = "Modéré"
        risk_color = "#F97316"
        risk_desc = "Risque cardiovasculaire modéré. Réduisez votre consommation de sel (< 5g par jour), pratiquez une activité physique régulière et surveillez votre tension régulièrement."
    else:
        risk_status = "Élevé"
        risk_color = "#EF4444"
        risk_desc = "Risque cardiovasculaire élevé. Forte probabilité de complications hypertensives (HTA, AVC ou Infarctus). Nous vous conseillons vivement de prendre rendez-vous dans le centre de santé le plus proche."

    st.markdown(f"""
    <div class="premium-card" style="text-align: center;">
        <span style="font-size: 13px; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Évaluation Clinique du Risque</span>
        <div style="font-size: 44px; font-weight: 800; color: {risk_color}; margin-top: 12px; margin-bottom: 8px; font-family: 'Outfit', sans-serif;">{risk_status}</div>
        <div style="font-size: 14px; color: #4B5563; margin-top: 20px; border-top: 1px solid #E2ECF2; padding-top: 15px; line-height: 1.5;">💡 <b>Recommandation officielle :</b> {risk_desc}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    st.warning("⚠️ **Rappel important :** Ce calculateur fournit une estimation basée sur les facteurs de risques déclarés et ne remplace en aucun cas un bilan médical complet (tension clinique au brassard, bilan lipidique, électrocardiogramme).")

elif page == "Fiche Patient":
    st.markdown('<div class="clean-title">📄 Votre Rapport de Consultation</div>', unsafe_allow_html=True)
    st.markdown('<div class="clean-subtitle">Téléchargez ou visualisez la synthèse clinique et l\'historique des échanges avec votre médecin virtuel.</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # Calculate status variables on-the-fly for rendering
    imc = st.session_state.poids / ((st.session_state.taille / 100.0) ** 2)
    if imc < 18.5:
        imc_status = "Maigreur"
        imc_color = "#EAB308"
    elif imc < 25.0:
        imc_status = "Normal"
        imc_color = "#22C55E"
    elif imc < 30.0:
        imc_status = "Surpoids"
        imc_color = "#F97316"
    else:
        imc_status = "Obésité"
        imc_color = "#EF4444"

    # Cardio Risk Status
    risk_score = 0
    if st.session_state.age >= 50: risk_score += 2
    if st.session_state.pas >= 140: risk_score += 3
    if st.session_state.tabac == "Oui": risk_score += 2

    if risk_score <= 1:
        risk_status = "Faible"
        risk_color = "#22C55E"
    elif risk_score <= 4:
        risk_status = "Modéré"
        risk_color = "#F97316"
    else:
        risk_status = "Élevé"
        risk_color = "#EF4444"

    # Compile patient report bytes
    pdf_bytes = generate_patient_pdf(imc, imc_status, risk_status, st.session_state.messages)

    logo_path = os.path.join(os.path.dirname(__file__), "tbibk_logo.png")
    b64_str = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

    card_html = f"""<div class="premium-card" style="margin-bottom: 25px;">
<div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #1E3A8A; padding-bottom: 12px; margin-bottom: 15px;">
<div style="text-align: left; font-size: 12px; color: #1E3A8A; line-height: 1.3;">
<b>ROYAUME DU MAROC</b><br/>
<span style="font-size: 10px; color: #475569;">Ministère de la Santé</span><br/>
<b><span style="font-size: 11px; color: #2563EB;">TBIBK — Assistant RAG</span></b>
</div>
<img src="data:image/png;base64,{b64_str}" style="width: 48px; height: 48px; object-fit: contain;">
<div style="text-align: right; font-size: 12px; color: #1E3A8A; line-height: 1.3; font-family: 'Tahoma', sans-serif;">
<b>المملكة المغربية</b><br/>
<span style="font-size: 10px; color: #475569;">وزارة الصحة والحماية الاجتماعية</span><br/>
<b><span style="font-size: 11px; color: #2563EB;">طبيبك — المساعد الطبي</span></b>
</div>
</div>
<h3 style="text-align: center; color: #1E3A8A; font-weight: 800; font-size: 15px; margin: 12px 0 18px 0; font-family: 'Outfit', 'Inter', sans-serif;">FICHE CLINIQUE DE CONSULTATION | بطاقة الاستشارة الطبية</h3>
<table style="width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; font-size: 13.5px;">
<thead>
<tr style="background-color: #F1F5F9; border-bottom: 2px solid #CBD5E1;">
<th style="padding: 10px; text-align: left; font-weight: 700; color: #0F172A;">Indicateur Évalué (المؤشر السريري)</th>
<th style="padding: 10px; text-align: right; font-weight: 700; color: #0F172A;">Résultat & Statut (النتيجة والتقييم)</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid #E2ECF2;">
<td style="padding: 10px; font-weight: 600; color: #334155;">Poids / Taille (الوزن والطول)</td>
<td style="padding: 10px; text-align: right; font-weight: 700; color: #0F172A;">{st.session_state.poids} kg / {st.session_state.taille} cm</td>
</tr>
<tr style="border-bottom: 1px solid #E2ECF2;">
<td style="padding: 10px; font-weight: 600; color: #334155;">Indice de Masse Corporelle (IMC / مؤشر كتلة الجسم)</td>
<td style="padding: 10px; text-align: right; font-weight: 700; color: {imc_color};">{imc:.1f} ({imc_status})</td>
</tr>
<tr>
<td style="padding: 10px; font-weight: 600; color: #334155;">Risque Cardiovasculaire (HTA / خطر القلب والضغط)</td>
<td style="padding: 10px; text-align: right; font-weight: 700; color: {risk_color};">{risk_status}</td>
</tr>
</tbody>
</table>
<div style="font-size: 11px; color: #64748B; text-align: center; margin-top: 20px; font-style: italic; border-top: 1px solid #F1F5F9; padding-top: 12px; line-height: 1.4;">
Avertissement : Ce document est une fiche d'information automatique et ne remplace pas une consultation médicale.<br/>
تنبيه: هذه البطاقة معلوماتية تم إنشاؤها تلقائياً ولا تعوض الاستشارة الطبية الفعلية.
</div>
</div>"""

    st.markdown(card_html, unsafe_allow_html=True)

    # Render download button
    st.download_button(
        label="📥 Télécharger ma Fiche Patient (PDF)",
        data=pdf_bytes,
        file_name="Fiche_Patient_Tbibk.pdf",
        mime="application/pdf",
        use_container_width=True
    )
