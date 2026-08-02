# -*- coding: utf-8 -*-
import os
import re
import ollama
from src.config import GENERATION_MODEL
from src.vector_store import SimpleVectorStore

# Keep the Ollama model loaded in RAM between requests (avoids cold-start reload latency)
OLLAMA_KEEP_ALIVE = "30m"

# Digits commonly used as Arabizi letter substitutes (3=ع, 7=ح, 9=ق).
# Only counts as a Darija signal when glued to letters (e.g. "3andi", "l7ala"),
# not when it's part of a plain number ("9 ans", "37 de fievre").
ARABIZI_DIGIT_PATTERN = re.compile(r'[a-zA-Z][379]|[379][a-zA-Z]')


class RAGPipeline:
    def __init__(self, model_name: str = GENERATION_MODEL):
        """
        Initializes the local single-step RAG Pipeline using Ollama.
        """
        self.model_name = model_name
        self.temperature = 0.2
        self.vector_store = SimpleVectorStore()

        # Load the index if it exists
        self.vector_store.load()

    def _get_client(self):
        """Returns the appropriate Ollama client (local default or custom host from OLLAMA_HOST/st.secrets)."""
        host = os.environ.get("OLLAMA_HOST")
        try:
            import streamlit as st
            if not host and "OLLAMA_HOST" in st.secrets:
                host = st.secrets["OLLAMA_HOST"]
        except Exception:
            pass
        if host:
            return ollama.Client(host=host)
        return ollama

    def _chat_complete(self, messages: list, temperature: float = 0.2, max_tokens: int = 500) -> str:
        """Executes a chat completion call via Groq API (with automatic retry and clean fallback)."""
        groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
        if not groq_api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    if "GROQ_API_KEY" in st.secrets:
                        groq_api_key = st.secrets["GROQ_API_KEY"]
                    elif "groq_api_key" in st.secrets:
                        groq_api_key = st.secrets["groq_api_key"]
            except Exception:
                pass

        if groq_api_key:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {str(groq_api_key).strip()}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Groq API HTTP error {resp.status_code}: {resp.text}")
                    if resp.status_code in [429, 500, 502, 503, 504]:
                        import time
                        time.sleep(1)
                        resp_retry = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20)
                        if resp_retry.status_code == 200:
                            return resp_retry.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Groq API exception: {str(e)}")

        try:
            client = self._get_client()
            res = client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                keep_alive=OLLAMA_KEEP_ALIVE
            )
            return res['message']['content'].strip()
        except Exception as e:
            print(f"Ollama local fallback unavailable: {str(e)}")
            raise RuntimeError("Le service IA est actuellement surchargé. Veuillez poser votre question à nouveau dans quelques secondes.")

    def set_model_name(self, model_name: str):
        """Updates the active generative model."""
        self.model_name = model_name

    def set_temperature(self, temperature: float):
        """Updates the active model generation temperature."""
        self.temperature = temperature

    def clean_and_pretranslate_darija(self, text: str) -> tuple:
        """
        Regex-based dictionary mapping of key Moroccan medical terms to French.
        Returns a tuple: (hint_category, direct_translation_if_any)
        """
        lower_text = text.lower()

        # Rule 1: Salt and Hypertension / Tension
        if ("mil" in lower_text or "sel" in lower_text or "ملح" in lower_text or "ملحة" in lower_text) and ("tansioun" in lower_text or "tansion" in lower_text or "tension" in lower_text or "طانسيو" in lower_text or "hta" in lower_text or "طانسون" in lower_text or "ضغط" in lower_text or "الضغط" in lower_text or "تانسيون" in lower_text):
            return ("hypertension artérielle (HTA)", "Comment réduire la consommation de sel pour faire baisser la tension artérielle (HTA) ?")

        # Rule 2: Diabetes / Skar diagnosis/check
        if ("skar" in lower_text or "sokar" in lower_text or "السكر" in lower_text or "سكر" in lower_text) and ("kifach" in lower_text or "كيفاش" in lower_text or "n9ed" in lower_text or "n9dar" in lower_text or "symptom" in lower_text or "a3rad" in lower_text or "أعراض" in lower_text or "اعراض" in lower_text or "نقدر" in lower_text or "تشخيص" in lower_text):
            return ("diabète", "Quels sont les symptômes du diabète et comment le diagnostiquer ?")

        # Rule 3: AVC Symptoms / Stroke
        if ("jalta" in lower_text or "جلطة" in lower_text or "الجلطة" in lower_text or "سكتة" in lower_text or "السكتة" in lower_text) and ("dimagh" in lower_text or "mokh" in lower_text or "دماغ" in lower_text or "مخ" in lower_text or "الدماغ" in lower_text or "المخ" in lower_text or "avc" in lower_text):
            return ("AVC (Accident Vasculaire Cérébral)", "Quels sont les symptômes d'un accident vasculaire cérébral (AVC) et comment réagir ?")

        # Rule 4: Crisis / Heart Attack
        if ("jalta" in lower_text or "جلطة" in lower_text or "الجلطة" in lower_text) and ("qalb" in lower_text or "قلب" in lower_text or "القلب" in lower_text or "infarctus" in lower_text or "crise" in lower_text or "صدر" in lower_text or "الصدر" in lower_text):
            return ("Infarctus du myocarde", "Quels sont les symptômes d'un infarctus du myocarde (crise cardiaque) ?")

        # Rule 5: Asthma in children
        if ("ddiq" in lower_text or "diiq" in lower_text or "الضيق" in lower_text or "ضيق" in lower_text or "asme" in lower_text or "asthme" in lower_text or "ربو" in lower_text or "الربو" in lower_text) and ("drari" in lower_text or "sghar" in lower_text or "ولد" in lower_text or "wld" in lower_text or "enfant" in lower_text or "أطفال" in lower_text or "اطفال" in lower_text or "طفل" in lower_text or "دراري" in lower_text or "صغار" in lower_text):
            return ("asthme chez l'enfant", "Comment prendre en charge et gérer une crise d'asthme chez l'enfant ?")

        # Rule 6: Contraceptives / safe ones
        if ("mawanih" in lower_text or "mawani3" in lower_text or "موانع" in lower_text or "مانع" in lower_text or "contracept" in lower_text) and ("haml" in lower_text or "حمل" in lower_text or "الحمل" in lower_text or "aamina" in lower_text or "safe" in lower_text or "آمنة" in lower_text or "امنة" in lower_text):
            return ("contraception", "Quels sont les moyens de contraception sûrs et efficaces selon le guide du Ministère de la Santé ?")

        # Rule 7: Pregnancy Nutrition
        if ("haml" in lower_text or "حمل" in lower_text or "الحمل" in lower_text or "grossesse" in lower_text) and ("makla" in lower_text or "ماكلة" in lower_text or "تغذية" in lower_text or "التغذية" in lower_text or "nassa" in lower_text or "نصائح" in lower_text or "نصيحة" in lower_text or "conseil" in lower_text):
            return ("grossesse et maternité", "Quels sont les conseils nutritionnels et d'alimentation pour une femme enceinte au Maroc ?")

        # General Category mappings (return category, no direct translation)
        if "skar" in lower_text or "sokar" in lower_text or "السكر" in lower_text or "سكر" in lower_text or "diabet" in lower_text:
            return ("diabète", "")
        if "tansioun" in lower_text or "tansion" in lower_text or "tansiot" in lower_text or "طانسيو" in lower_text or "hta" in lower_text or "ضغط" in lower_text or "الضغط" in lower_text:
            return ("hypertension artérielle (HTA)", "")
        if "jalta" in lower_text or "جلطة" in lower_text or "avc" in lower_text:
            return ("complications cardiovasculaires", "")
        if "ddiq" in lower_text or "diiq" in lower_text or "الضيق" in lower_text or "asthme" in lower_text:
            return ("asthme", "")
        if "mawanih" in lower_text or "mawani3" in lower_text or "contracept" in lower_text:
            return ("contraception", "")
        if "haml" in lower_text or "grossesse" in lower_text or "حمل" in lower_text:
            return ("grossesse et maternité", "")

        return ("", "")

    def detect_and_translate_query(self, user_query: str) -> dict:
        """
        Detects if query is Darija/Arabic or French.
        If it is Darija, translates/reformulates it to French using a guided local LLM prompt.
        """
        is_arabic = any(u'\u0600' <= c <= u'\u06FF' for c in user_query)
        is_arabizi = False
        if not is_arabic:
            darija_keywords = ["rani", "mrid", "bghit", "chkoun", "fayn", "kifach", "kifas", "m3a", "dyal", "dya", "hadi", "hada", "skar", "sokar", "sukhna", "sakhna", "tansioun", "tansiot", "jalta", "ddiq", "diiq", "hsasiya", "mawanih", "mawani3", "haml", "kina"]
            has_keyword = any(w in user_query.lower() for w in darija_keywords)
            # Only trust digit-as-letter substitution (3/7/9) when it's glued to
            # a letter, e.g. "3andi" or "l7ala" — NOT bare numbers like "9 ans"
            # or "37 de fievre", which would otherwise falsely flag French text.
            has_digit_letters = bool(ARABIZI_DIGIT_PATTERN.search(user_query))
            is_arabizi = has_keyword or has_digit_letters

        is_darija = is_arabic or is_arabizi
        script_type = "arabic" if is_arabic else ("latin" if is_arabizi else "french")

        query_fr = user_query
        if is_darija:
            # Get keyword hint and direct translation from clean_and_pretranslate_darija
            hint, direct_trans = self.clean_and_pretranslate_darija(user_query)

            if direct_trans:
                # Use the direct translation immediately and bypass LLM call!
                query_fr = direct_trans
            elif hint:
                # If there's a category hint but no direct translation, use guided LLM translation
                translation_prompt = f"""You are translating a medical question from Moroccan Darija/Arabizi to French.
The question is about: {hint}.
Translate the query: "{user_query}" into a simple, direct French medical question about {hint}.
Output ONLY the French translation, no other text."""
                query_fr = self._safe_translate(translation_prompt, fallback=query_fr)
            else:
                # General fallback translation
                translation_prompt = f"""You are a professional medical translator translating from Moroccan Darija/Arabizi to French.
Translate this question directly into a clear French medical question:
"{user_query}"

Rules:
1. Translate accurately.
2. Output ONLY the French translation. Do not include any explanations, notes, or metadata.
3. Be direct.

Translation:"""
                query_fr = self._safe_translate(translation_prompt, fallback=query_fr)

        return {
            "is_darija": is_darija,
            "script_type": script_type,
            "query_fr": query_fr
        }

    def _safe_translate(self, prompt: str, fallback: str) -> str:
        """Runs a translation prompt through LLM (Groq / Ollama), falling back to original text on error."""
        try:
            translated = self._chat_complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            translated = translated.replace('"', '').replace("'", "")
            return translated if translated else fallback
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return fallback

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Detects CJK ideographs that small multilingual LLMs (e.g. Qwen 1.5B) sometimes leak into Arabic output."""
        return bool(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text))

    @staticmethod
    def _strip_cjk(text: str) -> str:
        """Removes leaked CJK characters/punctuation and cleans up residual fragments."""
        cleaned = re.sub(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff，。、；：？！]+', '', text)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_prompt_echo(text: str) -> str:
        """
        Strips leading boilerplate the model may echo back (e.g. the translation
        prompt's label) by cutting everything before the first Arabic letter,
        rather than matching one exact hardcoded string.
        """
        match = re.search(r'[\u0600-\u06FF]', text)
        if match:
            return text[match.start():].strip()
        return text.strip()

    def is_pure_greeting(self, text: str) -> bool:
        """
        Detects if a user message is a simple greeting (salam, hello, hi, etc.)
        and should bypass RAG retrieval.
        """
        cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = cleaned.split()

        if not words or len(words) > 3:
            return False

        greeting_keywords = {
            "salam", "salut", "hello", "hi", "hey", "bonjour", "bonsoir", "yo",
            "salamo3alikom", "salamo3alaykom", "salamo3likom", "salamoualikoum", "salamoualaykoum",
            "sbah", "lkhir", "lkheir", "msa", "msah", "labas", "cava",
            "سلام", "السلام", "علكم", "عليكم", "صباح", "الخير", "مساء", "لاباس"
        }

        return all(word in greeting_keywords for word in words)

    def generate_response(self, original_query: str, query_analysis: dict, retrieved_chunks: list) -> str:
        """
        Generates the final response based on the retrieved French chunks using Ollama.
        Applies strict constraints to avoid hallucinations and responds in the correct
        language/script (French, Arabic script, or Arabizi/Latin script).
        """
        context_str = ""
        for i, (chunk, score) in enumerate(retrieved_chunks[:3]):
            chunk_text = chunk['text'][:1200]
            context_str += f"--- Source {i+1} (Fichier: {chunk['source']}, Page: {chunk.get('page')}) ---\n"
            context_str += f"{chunk_text}\n\n"

        is_darija = query_analysis.get("is_darija", False)
        script_type = query_analysis.get("script_type", "french")
        query_fr = query_analysis.get("query_fr", original_query)

        NO_INFO_TOKEN = "NO_INFO"

        if is_darija:
            system_prompt_ar = f"""Vous êtes Tbibk (طبيبك), l'assistant médical officiel d'information du Ministère de la Santé du Maroc.
Votre rôle est de répondre directement en langue Arabe (اللغة العربية) de manière professionnelle, bienveillante et médicalement exacte à la question médicale de l'utilisateur.

DIRECTIVES :
1. Répondez en Arabe clair et lisible.
2. Commencez impérativement votre réponse par: "الجواب الرسمي من وزارة الصحة المغربية هو: "
3. Expliquez les symptômes clés et conseils pratiques d'urgence ou de prévention basés sur le contexte ci-dessous.

--- CONTEXTE OFFICIEL DE SANTÉ ---
{context_str}
"""
            try:
                res_ar = self._chat_complete(
                    messages=[
                        {"role": "system", "content": system_prompt_ar},
                        {"role": "user", "content": f"Question médicale : {query_fr}"}
                    ],
                    temperature=0.2,
                    max_tokens=650
                )

                if not res_ar or res_ar.strip() == NO_INFO_TOKEN:
                    return self._no_info_message(script_type)

                if self._contains_cjk(res_ar):
                    res_ar = self._strip_cjk(res_ar)

                prefix = "الجواب الرسمي من وزارة الصحة المغربية هو: "
                if not res_ar.startswith("الجواب الرسمي"):
                    res_ar = f"{prefix}\n{res_ar}"

                return res_ar

            except Exception as e:
                print(f"Error in Darija response generation: {str(e)}")
                return self._no_info_message(script_type)
        else:
            # Classic French generation
            system_prompt = f"""Vous êtes Tbibk (طبيبك), l'assistant médical officiel d'information du Ministère de la Santé du Maroc.
Votre rôle est de répondre de manière professionnelle, claire et médicalement précise en Français.

DIRECTIVES :
1. Synthétisez les recommandations médicales officielles à partir du contexte ci-dessous.
2. Fournissez des explications claires et bienveillantes.

--- CONTEXTE OFFICIEL DE SANTÉ ---
{context_str}
"""
            assistant_prefix = "Selon les documents officiels du Ministère de la Santé : "
            try:
                generated_content = self._chat_complete(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": original_query}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                return f"{assistant_prefix}{generated_content}"
            except Exception as e:
                print(f"Error during response generation locally: {str(e)}")
                return f"Error: Failed to generate response ({str(e)})"

    def _translate_to_arabic_and_prefix(self, res_fr: str) -> str:
        """Translates a French answer to Arabic script and applies the official-source prefix."""
        translation_to_ar_prompt = f"""ترجم النص الطبي التالي بدقة إلى اللغة العربية الفصحى البسيطة والواضحة. أجب فقط بالترجمة العربية ولا تضف أي نص آخر. ممنوع منعا تاما استخدام الأحرف الصينية أو أي لغة أخرى غير العربية.

تنبيه ترجمة المصطلحات:
- sel / sodium -> الملح / الصوديوم
- sucre / glycémie -> السكر / الجلوكوز

النص بالفرنسية:
{res_fr}

الترجمة باللغة العربية:"""

        res_ar = ""
        for _attempt in range(2):
            try:
                res_ar = self._chat_complete(
                    messages=[{"role": "user", "content": translation_to_ar_prompt}],
                    temperature=0.0,
                    max_tokens=600
                )
            except Exception as e:
                print(f"Error in AR translation: {str(e)}")
                break
            if not self._contains_cjk(res_ar):
                break
        if self._contains_cjk(res_ar):
            res_ar = self._strip_cjk(res_ar)

        res_ar = self._strip_prompt_echo(res_ar)

        # Guardrail: the small model sometimes confuses sel/sodium with sucre in
        # translation. This patch only covers that one known confusion — it is
        # not a substitute for validating other terminology (dosages, negations,
        # medication names), which should be handled via stronger prompting.
        if ("sel" in res_fr.lower() or "sodium" in res_fr.lower()) and "sucre" not in res_fr.lower():
            res_ar = (res_ar
                      .replace("السكر", "الملح")
                      .replace("السكري", "الملحي")
                      .replace("أطعمة سكرية", "أطعمة مالحة")
                      .replace("الماء السكري", "الماء"))

        assistant_prefix = "الجواب الرسمي من وزارة الصحة المغربية هو: "
        return f"{assistant_prefix}{res_ar}"

    @staticmethod
    def _no_info_message(script_type: str) -> str:
        """Returns the 'no info found' message in the script matching the user's input."""
        if script_type == "arabic":
            return "ما عنديش هاد المعلومة فالمصادر الرسمية ديال وزارة الصحة."
        elif script_type == "latin":
            return "Ma 3ndich had l'ma3loma f les sources rasmiya dyal Wizarat Assiha."
        else:
            return "Désolé, je ne dispose pas d'informations officielles à ce sujet dans les documents du Ministère de la Santé."

    def answer_query(self, user_query: str, top_k: int = 5) -> dict:
        """
        Executes the entire local RAG pipeline:
        1. Checks if it is a pure greeting (bypasses RAG)
        2. Translation/Analysis of query
        3. Vector search in French document collection using local embeddings
        4. Response generation using local Ollama model
        Returns a dict with: 'response', 'retrieved_sources', 'query_fr', 'is_darija', 'script_type'
        """
        if self.vector_store.is_empty():
            return {
                "response": "La base de connaissances est vide. Indexation automatique en cours dans le thread d'arrière-plan...",
                "retrieved_sources": [],
                "query_fr": user_query,
                "is_darija": False,
                "script_type": "french"
            }

        # 1. Check if it is a pure greeting
        if self.is_pure_greeting(user_query):
            analysis = self.detect_and_translate_query(user_query)
            is_arabic = (analysis.get("script_type") == "arabic")
            is_latin = (analysis.get("script_type") == "latin")

            if is_arabic:
                sys_prompt = "أنت طبيبك (Tbibk)، مساعد طبي افتراضي رسمي لوزارة الصحة المغربية. رحب بالمستخدم بلطف بالدارجة المغربية (بالحروف العربية) واسأله كيف يمكنك مساعدته اليوم في صحته."
            elif is_latin:
                sys_prompt = "You are Tbibk, an official virtual medical assistant for the Moroccan Ministry of Health. Greet the user warmly in Moroccan Darija (Arabizi/latin script, e.g. using 3, 7, 9) and ask how you can help them with their health today."
            else:
                sys_prompt = "Vous êtes Tbibk, un assistant médical virtuel officiel du Ministère de la Santé du Maroc. Greet the user warmly in French and ask how you can help them with their health today."

            try:
                response_text = self._chat_complete(
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.5,
                    max_tokens=100
                )
            except Exception as e:
                print(f"Error during greeting response: {str(e)}")
                response_text = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"

            return {
                "response": response_text,
                "retrieved_sources": [],
                "query_fr": user_query,
                "is_darija": analysis.get("is_darija", False),
                "script_type": analysis.get("script_type", "french")
            }

        # 2. Analyze query language and translate to French if Darija
        analysis = self.detect_and_translate_query(user_query)

        # 3. Search Vector Store using the translated French query
        results = self.vector_store.search(analysis["query_fr"], top_k=top_k)

        # 4. Generate response using local model
        response_text = self.generate_response(user_query, analysis, results)

        return {
            "response": response_text,
            "retrieved_sources": results,
            "query_fr": analysis.get("query_fr", user_query),
            "is_darija": analysis.get("is_darija", False),
            "script_type": analysis.get("script_type", "french")
        }