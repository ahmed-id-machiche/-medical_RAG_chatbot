# -*- coding: utf-8 -*-
import os
import uuid
import base64
from io import BytesIO

import dotenv
dotenv.load_dotenv()

import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.parser import load_and_chunk_all_documents
from src.downloader import create_fallback_documents

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CACHED RESOURCES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tbibk",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def get_rag_pipeline():
    """Cache the heavy RAGPipeline instance once across all Streamlit sessions."""
    return RAGPipeline()

pipeline = get_rag_pipeline()

# -----------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "is_indexing" not in st.session_state:
    st.session_state.is_indexing = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_id" not in st.session_state:
    st.session_state.conv_id = ""
if "my_conversations" not in st.session_state:
    st.session_state.my_conversations = {}

# Clinical default inputs
for key, val in [("poids", 70.0), ("taille", 170.0), ("age", 45), ("sexe", "Femme"), ("pas", 120), ("tabac", "Non")]:
    if key not in st.session_state:
        st.session_state[key] = val

def check_and_build_index():
    if pipeline.vector_store.is_empty():
        st.session_state.is_indexing = True
        try:
            print("--- Auto-Indexing Medical Knowledge Base ---")
            create_fallback_documents()
            chunks = load_and_chunk_all_documents()
            if chunks:
                pipeline.vector_store.reset_collection()
                pipeline.vector_store.add_chunks(chunks)
                pipeline.vector_store.save()
        except Exception as e:
            print(f"Auto-indexing warning: {str(e)}")
        finally:
            st.session_state.is_indexing = False

if "indexing_started" not in st.session_state:
    st.session_state.indexing_started = True
    check_and_build_index()

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS (CALCULATORS & SESSION CONVERSATIONS)
# -----------------------------------------------------------------------------
def get_base64_image(file_path):
    target_path = os.path.join("assets", file_path) if os.path.exists(os.path.join("assets", file_path)) else file_path
    if os.path.exists(target_path):
        with open(target_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

def get_logo_path():
    asset_logo = os.path.join("assets", "tbibk_logo.png")
    return asset_logo if os.path.exists(asset_logo) else "tbibk_logo.png"

# -----------------------------------------------------------------------------
# 4. CONSOLIDATED UNIFIED CSS STYLING
# -----------------------------------------------------------------------------
chat_icon_b64 = get_base64_image("3d_chat_icon.png")
imc_icon_b64 = get_base64_image("3d_imc_icon.png")
heart_icon_b64 = get_base64_image("3d_heart_icon.png")
report_icon_b64 = get_base64_image("3d_report_icon.png")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp, .main, [data-testid="stMain"], [data-testid="stMainViewContainer"], [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {{
        background-color: #549FC4 !important;
        background-image: none !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Force stBottom and all its wrappers/children to be blue like the rest of the page */
    [data-testid="stBottom"],
    [data-testid="stBottom"] *,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] *,
    div[class*="stBottom"],
    div[class*="stBottom"] *,
    div[data-testid="stForm"],
    footer,
    footer * {{
        background-color: #549FC4 !important;
        background: #549FC4 !important;
        background-image: none !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Preserve chat input container as crisp white */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] * {{
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        background: transparent !important;
        color: #111827 !important;
    }}
    
    .block-container {{ max-width: 800px !important; padding-top: 2rem !important; margin: 0 auto !important; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid #488EAF !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] div {{ color: #FFFFFF !important; }}
    
    /* Hide Radio Nav circle dots aggressively */
    div[role="radiogroup"] label input[type="radio"],
    div[role="radiogroup"] label svg,
    div[role="radiogroup"] label [role="presentation"],
    div[role="radiogroup"] label > div:first-child:not(:last-child) {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }}
    div[role="radiogroup"] label {{ background-color: transparent !important; border-radius: 8px !important; padding: 10px 14px !important; margin-bottom: 6px !important; cursor: pointer !important; display: flex !important; align-items: center !important; }}
    div[role="radiogroup"] label p {{ font-family: 'Outfit', sans-serif !important; font-size: 17px !important; font-weight: 600 !important; color: #FFFFFF !important; margin: 0 !important; }}
    div[role="radiogroup"] label:nth-of-type(1)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{chat_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:nth-of-type(2)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{imc_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:nth-of-type(3)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{heart_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:nth-of-type(4)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{report_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:has(input:checked) {{ background-color: #FFFFFF !important; }}
    div[role="radiogroup"] label:has(input:checked) p {{ color: #549FC4 !important; font-weight: 700 !important; }}
    
    [data-testid="stSidebar"] button {{ background-color: rgba(255, 255, 255, 0.1) !important; border: 1px dashed rgba(255, 255, 255, 0.6) !important; font-weight: 600 !important; color: #FFFFFF !important; border-radius: 8px !important; }}
    .clean-title {{ font-size: 38px !important; font-weight: 700 !important; color: #FFFFFF !important; text-align: center !important; font-family: 'Outfit', sans-serif !important; margin-bottom: 10px !important; }}
    .clean-subtitle {{ font-size: 14.5px !important; color: rgba(255, 255, 255, 0.85) !important; text-align: center !important; margin-bottom: 25px !important; }}
    .stChatMessage {{ background-color: #FFFFFF !important; border-radius: 12px !important; padding: 16px 20px !important; margin-bottom: 12px !important; border: 1px solid #E2ECF2 !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important; }}
    .stChatMessage p, .stChatMessage span, .stChatMessage div {{ color: #1F2937 !important; }}
    .lang-badge {{ font-size: 10px; padding: 2px 6px; border-radius: 4px; color: #4B5563; background-color: #F3F4F6; border: 1px solid #E5E7EB; font-weight: 600; display: inline-block; margin-bottom: 6px; }}
    .reformulation-note {{ font-size: 11px; color: #6B7280; margin-bottom: 6px; font-style: italic; }}
    .arabic-text {{ direction: rtl; text-align: right; font-family: 'Segoe UI', sans-serif; line-height: 1.7; font-size: 15.5px; color: #111827 !important; }}
    .premium-card {{ background: #FFFFFF !important; border-radius: 16px !important; padding: 25px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important; }}
    .premium-card p, .premium-card td, .premium-card th {{ color: #1F2937 !important; }}
    div[data-testid="stChatInput"] {{ border: 1px solid #E5E7EB !important; border-radius: 26px !important; background-color: #FFFFFF !important; }}
    div[data-testid="stChatInput"] textarea {{ color: #111827 !important; }}
</style>
""", unsafe_allow_html=True)

def save_current_conversation():
    if "conv_id" not in st.session_state or not st.session_state.conv_id:
        st.session_state.conv_id = str(uuid.uuid4())
    if not st.session_state.messages:
        return
    first_msg = st.session_state.messages[0]["content"]
    title = first_msg[:24] + "..." if len(first_msg) > 24 else first_msg
    st.session_state.my_conversations[st.session_state.conv_id] = {
        "id": st.session_state.conv_id,
        "title": title,
        "messages": list(st.session_state.messages),
        "poids": st.session_state.get("poids", 70.0),
        "taille": st.session_state.get("taille", 170.0),
        "age": st.session_state.get("age", 45),
        "sexe": st.session_state.get("sexe", "Femme"),
        "pas": st.session_state.get("pas", 120),
        "tabac": st.session_state.get("tabac", "Non")
    }

def list_conversations():
    if "my_conversations" not in st.session_state:
        return []
    return list(st.session_state.my_conversations.values())

def load_conversation(conv_id):
    if "my_conversations" in st.session_state and conv_id in st.session_state.my_conversations:
        data = st.session_state.my_conversations[conv_id]
        st.session_state.conv_id = data["id"]
        st.session_state.messages = list(data["messages"])
        st.session_state.poids = data.get("poids", 70.0)
        st.session_state.taille = data.get("taille", 170.0)
        st.session_state.age = data.get("age", 45)
        st.session_state.sexe = data.get("sexe", "Femme")
        st.session_state.pas = data.get("pas", 120)
        st.session_state.tabac = data.get("tabac", "Non")

def calculate_imc(poids, taille):
    imc = poids / ((taille / 100.0) ** 2)
    if imc < 18.5:
        return imc, "Maigreur", "#EAB308", "Poids insuffisant. Veillez à enrichir votre alimentation avec des aliments denses en nutriments."
    elif imc < 25.0:
        return imc, "Normal", "#22C55E", "Poids idéal. Vous êtes dans la plage de santé recommandée. Continuez ainsi !"
    elif imc < 30.0:
        return imc, "Surpoids", "#F97316", "Surpoids léger. Limitez les graisses, les sucres raffinés, et faites au moins 30 minutes d'activité physique par jour."
    else:
        return imc, "Obésité", "#EF4444", "Obésité. Risque accru de maladies cardiovasculaires, d'hypertension et de diabète. Nous vous conseillons de consulter votre centre de santé."

def calculate_cardio_risk(age, pas, tabac):
    score = 0
    if age >= 50: score += 2
    if pas >= 140: score += 3
    if tabac == "Oui": score += 2
    if score <= 1:
        return "Faible", "#22C55E", "Risque cardiovasculaire faible. Continuez à maintenir une alimentation saine et équilibrée."
    elif score <= 4:
        return "Modéré", "#F97316", "Risque cardiovasculaire modéré. Réduisez votre consommation de sel (< 5g par jour), pratiquez une activité physique régulière."
    else:
        return "Élevé", "#EF4444", "Risque cardiovasculaire élevé. Forte probabilité de complications hypertensives (HTA, AVC). Prenez rendez-vous dans le centre de santé le plus proche."

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

def generate_patient_pdf(imc, imc_status, risk_status, messages):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='PatientTitle', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=colors.HexColor("#549FC4"), alignment=1, spaceAfter=20)
    header_style = ParagraphStyle(name='PatientHeader', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor("#0F172A"), spaceBefore=14, spaceAfter=8, keepWithNext=True)
    body_style = ParagraphStyle(name='PatientBody', fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor("#334155"), spaceAfter=6)
    bold_style = ParagraphStyle(name='PatientBold', fontName='Helvetica-Bold', fontSize=10, leading=14, textColor=colors.HexColor("#0F172A"), spaceAfter=6)

    story = []
    
    # Header logo if available
    logo_path = get_logo_path()
    if os.path.exists(logo_path):
        try:
            img = RLImage(logo_path, width=65, height=65)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 10))
        except Exception:
            pass

    story.extend([
        Paragraph("Tbibk - Fiche Clinique de Consultation", title_style),
        Spacer(1, 10),
        Paragraph("1. Synthèse des Paramètres Cliniques", header_style)
    ])
    param_data = [
        [Paragraph("<b>Indicateur évalué</b>", bold_style), Paragraph("<b>Statut clinique</b>", bold_style)],
        [Paragraph("Indice de Masse Corporelle (IMC)", body_style), Paragraph(f"{imc:.1f} ({imc_status})", body_style)],
        [Paragraph("Risque Cardio-Vasculaire (Score HTA)", body_style), Paragraph(risk_status, body_style)]
    ]
    t = Table(param_data, colWidths=[250, 250])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F8FAFC")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BOTTOMPADDING', (0,0), (-1,-1), 6), ('TOPPADDING', (0,0), (-1,-1), 6)]))
    story.extend([t, Spacer(1, 15), Paragraph("2. Historique de la Discussion Médicale", header_style)])
    
    if not messages:
        story.append(Paragraph("Aucune discussion enregistrée dans cette session.", body_style))
    else:
        for msg in messages:
            role = "Patient" if msg["role"] == "user" else "Tbibk"
            story.append(Paragraph(f"<b>{role} :</b>", bold_style))
            clean_content = msg["content"].replace("<br/>", "\n").replace("<b>", "").replace("</b>", "").replace("<div class=\"arabic-text\">", "").replace("</div>", "")
            story.append(Paragraph(clean_content, body_style))
            story.append(Spacer(1, 4))
            
    story.extend([Spacer(1, 20), Paragraph("<i>Avertissement : Ce document est une fiche d'information automatique et ne remplace pas une consultation médicale.</i>", body_style)])
    doc.build(story)
    return buffer.getvalue()

# -----------------------------------------------------------------------------
# 4. CONSOLIDATED UNIFIED CSS STYLING
# -----------------------------------------------------------------------------
chat_icon_b64 = get_base64_image("3d_chat_icon.png")
imc_icon_b64 = get_base64_image("3d_imc_icon.png")
heart_icon_b64 = get_base64_image("3d_heart_icon.png")
report_icon_b64 = get_base64_image("3d_report_icon.png")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp, .main, [data-testid="stMain"], [data-testid="stMainViewContainer"], [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stSidebar"] {{
        background-color: #549FC4 !important;
        background-image: none !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }}
    .block-container {{ max-width: 800px !important; padding-top: 2rem !important; margin: 0 auto !important; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid #488EAF !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] div {{ color: #FFFFFF !important; }}
    
    /* Radio Nav without circles */
    div[role="radiogroup"] label input[type="radio"], div[role="radiogroup"] label svg {{ display: none !important; }}
    div[role="radiogroup"] label {{ background-color: transparent !important; border-radius: 8px !important; padding: 10px 14px !important; margin-bottom: 6px !important; cursor: pointer !important; display: flex !important; align-items: center !important; }}
    div[role="radiogroup"] label p {{ font-family: 'Outfit', sans-serif !important; font-size: 17px !important; font-weight: 600 !important; color: #FFFFFF !important; margin: 0 !important; }}
    div[role="radiogroup"] label:nth-of-type(1)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{chat_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:nth-of-type(2)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{imc_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:nth-of-type(3)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{heart_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:nth-of-type(4)::before {{ content: "" !important; display: inline-block !important; width: 28px !important; height: 28px !important; margin-right: 12px !important; background-image: url("data:image/png;base64,{report_icon_b64}") !important; background-size: contain !important; background-repeat: no-repeat !important; }}
    div[role="radiogroup"] label:has(input:checked) {{ background-color: #FFFFFF !important; }}
    div[role="radiogroup"] label:has(input:checked) p {{ color: #549FC4 !important; font-weight: 700 !important; }}
    
    [data-testid="stSidebar"] button {{ background-color: rgba(255, 255, 255, 0.1) !important; border: 1px dashed rgba(255, 255, 255, 0.6) !important; font-weight: 600 !important; color: #FFFFFF !important; border-radius: 8px !important; }}
    .clean-title {{ font-size: 38px !important; font-weight: 700 !important; color: #FFFFFF !important; text-align: center !important; font-family: 'Outfit', sans-serif !important; margin-bottom: 10px !important; }}
    .clean-subtitle {{ font-size: 14.5px !important; color: rgba(255, 255, 255, 0.85) !important; text-align: center !important; margin-bottom: 25px !important; }}
    .stChatMessage {{ background-color: #FFFFFF !important; border-radius: 12px !important; padding: 16px 20px !important; margin-bottom: 12px !important; border: 1px solid #E2ECF2 !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important; }}
    .stChatMessage p, .stChatMessage span, .stChatMessage div {{ color: #1F2937 !important; }}
    .lang-badge {{ font-size: 10px; padding: 2px 6px; border-radius: 4px; color: #4B5563; background-color: #F3F4F6; border: 1px solid #E5E7EB; font-weight: 600; display: inline-block; margin-bottom: 6px; }}
    .reformulation-note {{ font-size: 11px; color: #6B7280; margin-bottom: 6px; font-style: italic; }}
    .arabic-text {{ direction: rtl; text-align: right; font-family: 'Segoe UI', sans-serif; line-height: 1.7; font-size: 15.5px; color: #111827 !important; }}
    .premium-card {{ background: #FFFFFF !important; border-radius: 16px !important; padding: 25px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important; }}
    .premium-card p, .premium-card td, .premium-card th {{ color: #1F2937 !important; }}
    div[data-testid="stChatInput"] {{ border: 1px solid #E5E7EB !important; border-radius: 26px !important; background-color: #FFFFFF !important; }}
    div[data-testid="stChatInput"] textarea {{ color: #111827 !important; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & DISCUSSIONS
# -----------------------------------------------------------------------------
logo_path = get_logo_path()

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.markdown("<div style='padding-top: 15px; margin-bottom: 10px;'><span style='font-size: 20px; font-weight: 700;'>🏥 Tbibk Workspace</span></div>", unsafe_allow_html=True)

if st.sidebar.button("➕ Nouvelle Discussion", use_container_width=True):
    st.session_state.conv_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.navigation_page = "Chatbot Tbibk"
    st.rerun()

page = st.sidebar.radio(
    "Menu :",
    ["Chatbot Tbibk", "Calculateur d'IMC", "Risque Cardiovasculaire", "Fiche Patient"],
    key="navigation_page"
)

st.sidebar.markdown("<hr style='margin: 15px 0; border-color: rgba(255, 255, 255, 0.2);'/>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='font-size: 11px; font-weight: 700; opacity: 0.8;'>🕒 Discussions passées</span>", unsafe_allow_html=True)

past_convs = list_conversations()
if not past_convs:
    st.sidebar.markdown("<span style='font-size: 12px; opacity: 0.6; font-style: italic;'>Aucun historique</span>", unsafe_allow_html=True)
else:
    for conv in past_convs:
        if st.sidebar.button(f"💬 {conv['title']}", key=f"conv_{conv['id']}", use_container_width=True):
            load_conversation(conv['id'])
            st.session_state.navigation_page = "Chatbot Tbibk"
            st.rerun()

# -----------------------------------------------------------------------------
# 6. MAIN PAGE VIEWS
# -----------------------------------------------------------------------------
if page == "Chatbot Tbibk":
    logo_col1, logo_col2, logo_col3 = st.columns([1.5, 1, 1.5])
    with logo_col2:
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)

    st.markdown("<div style='text-align: center; margin-top: -5px; margin-bottom: 20px;'><span style='font-size: 34px; font-weight: 800; color: #FFFFFF; font-family: \"Outfit\", sans-serif;'>TBIBK</span></div>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("<div style='text-align: center; margin-bottom: 25px;'><span style='font-size: 24px; font-weight: 600; opacity: 0.9;'>Comment puis-je vous aider ?</span></div>", unsafe_allow_html=True)

    st.warning("⚠️ **Avertissement :** Tbibk est un assistant d'information médicale basé sur les sources officielles du Ministère de la Santé du Maroc. Il ne remplace pas l'avis d'un professionnel de santé.")

    if st.session_state.is_indexing:
        st.info("⚡ **Indexation en cours :** La base de données locale se charge en arrière-plan...")

    for msg in st.session_state.messages:
        avatar_icon = "👤" if msg["role"] == "user" else logo_path
        with st.chat_message(msg["role"], avatar=avatar_icon):
            if msg["role"] == "assistant":
                is_darija = msg.get("is_darija", False)
                badge = '<div class="lang-badge">Darija 🇲🇦</div>' if is_darija else '<div class="lang-badge">Français 🇫🇷</div>'
                st.markdown(badge, unsafe_allow_html=True)
                if is_darija and msg.get("query_fr"):
                    st.markdown(f'<div class="reformulation-note">Question traduite : "{msg["query_fr"]}"</div>', unsafe_allow_html=True)
                if is_darija:
                    st.markdown(f'<div class="arabic-text">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Posez votre question médicale..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_current_conversation()
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant", avatar=logo_path):
            with st.spinner("Recherche dans les sources officielles..."):
                try:
                    res = pipeline.answer_query(user_query, top_k=3)
                    sources_metadata = [{"source": c["source"], "page": c.get("page", "N/A"), "text": c["text"]} for c, s in res["retrieved_sources"]]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["response"],
                        "is_darija": res["is_darija"],
                        "query_fr": res["query_fr"],
                        "sources": sources_metadata
                    })
                    save_current_conversation()
                    st.rerun()
                except Exception as e:
                    st.error(f"Désolé, une erreur est survenue : {str(e)}")

elif page == "Calculateur d'IMC":
    st.markdown('<div class="clean-title">⚖️ Indice de Masse Corporelle (IMC)</div>', unsafe_allow_html=True)
    st.markdown('<div class="clean-subtitle">Évaluez votre poids par rapport à votre taille selon les directives du Ministère de la Santé du Maroc.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        poids = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=st.session_state.poids, step=1.0)
        st.session_state.poids = poids
    with col2:
        taille = st.number_input("Taille (cm)", min_value=100.0, max_value=220.0, value=st.session_state.taille, step=1.0)
        st.session_state.taille = taille
    save_current_conversation()
        
    imc, imc_status, imc_color, imc_desc = calculate_imc(poids, taille)
    st.markdown(f"""
    <div class="premium-card" style="text-align: center;">
        <span style="font-size: 13px; color: #475569; font-weight: 700; text-transform: uppercase;">Votre Résultat d'IMC</span>
        <div style="font-size: 54px; font-weight: 800; color: {imc_color}; margin-top: 12px; font-family: 'Outfit', sans-serif;">{imc:.1f}</div>
        <div style="font-size: 20px; font-weight: 700; color: {imc_color}; font-family: 'Outfit', sans-serif;">{imc_status}</div>
        <div style="font-size: 14px; color: #4B5563; margin-top: 20px; border-top: 1px solid #E2ECF2; padding-top: 15px;">💡 <b>Conseil :</b> {imc_desc}</div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Risque Cardiovasculaire":
    st.markdown('<div class="clean-title">💓 Risque Cardio-Vasculaire & HTA</div>', unsafe_allow_html=True)
    st.markdown('<div class="clean-subtitle">Évaluez votre risque cardiovasculaire selon les normes cliniques officielles du Maroc.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Âge", min_value=18, max_value=100, value=st.session_state.age)
        sexe = st.selectbox("Sexe", ["Femme", "Homme"], index=0 if st.session_state.sexe == "Femme" else 1)
        st.session_state.age = age
        st.session_state.sexe = sexe
    with col2:
        pas = st.slider("Tension Systolique (PAS en mmHg)", min_value=80, max_value=220, value=st.session_state.pas)
        tabac = st.selectbox("Fumeur", ["Non", "Oui"], index=0 if st.session_state.tabac == "Non" else 1)
        st.session_state.pas = pas
        st.session_state.tabac = tabac
    save_current_conversation()
        
    risk_status, risk_color, risk_desc = calculate_cardio_risk(age, pas, tabac)
    st.markdown(f"""
    <div class="premium-card" style="text-align: center;">
        <span style="font-size: 13px; color: #475569; font-weight: 700; text-transform: uppercase;">Évaluation Clinique du Risque</span>
        <div style="font-size: 44px; font-weight: 800; color: {risk_color}; margin-top: 12px; font-family: 'Outfit', sans-serif;">{risk_status}</div>
        <div style="font-size: 14px; color: #4B5563; margin-top: 20px; border-top: 1px solid #E2ECF2; padding-top: 15px;">💡 <b>Recommandation :</b> {risk_desc}</div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Fiche Patient":
    st.markdown('<div class="clean-title">📄 Votre Rapport de Consultation</div>', unsafe_allow_html=True)
    st.markdown('<div class="clean-subtitle">Téléchargez la synthèse clinique et l\'historique des échanges.</div>', unsafe_allow_html=True)

    imc, imc_status, imc_color, _ = calculate_imc(st.session_state.poids, st.session_state.taille)
    risk_status, risk_color, _ = calculate_cardio_risk(st.session_state.age, st.session_state.pas, st.session_state.tabac)
    pdf_bytes = generate_patient_pdf(imc, imc_status, risk_status, st.session_state.messages)

    st.markdown(f"""
    <div class="premium-card" style="margin-bottom: 25px;">
        <span style="font-size: 13px; color: #475569; font-weight: 700; text-transform: uppercase; text-align: center; display: block; margin-bottom: 15px;">Résumé Médical du Patient</span>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #E2ECF2;"><td style="padding: 10px 0;">Poids / Taille :</td><td style="text-align: right; font-weight: 700;">{st.session_state.poids} kg / {st.session_state.taille} cm</td></tr>
            <tr style="border-bottom: 1px solid #E2ECF2;"><td style="padding: 10px 0;">IMC :</td><td style="text-align: right; font-weight: 700; color: {imc_color};">{imc:.1f} ({imc_status})</td></tr>
            <tr><td style="padding: 10px 0;">Risque Cardiovasculaire / HTA :</td><td style="text-align: right; font-weight: 700; color: {risk_color};">{risk_status}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label="📥 Télécharger ma Fiche Patient (PDF)",
        data=pdf_bytes,
        file_name="Fiche_Patient_Tbibk.pdf",
        mime="application/pdf",
        use_container_width=True
    )
