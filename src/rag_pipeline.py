# -*- coding: utf-8 -*-
import os
import re
import ollama
from src.config import GENERATION_MODEL
from src.vector_store import SimpleVectorStore

# Keep the Ollama model loaded in RAM between requests (avoids cold-start reload latency)
OLLAMA_KEEP_ALIVE = "30m"

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
        if ("mil" in lower_text or "sel" in lower_text or "ملح" in lower_text) and ("tansioun" in lower_text or "tansion" in lower_text or "tension" in lower_text or "طانسيو" in lower_text or "hta" in lower_text or "طانسون" in lower_text):
            return ("hypertension artérielle (HTA)", "Comment réduire la consommation de sel pour faire baisser la tension artérielle (HTA) ?")
            
        # Rule 2: Diabetes / Skar diagnosis/check
        if ("skar" in lower_text or "sokar" in lower_text or "السكر" in lower_text) and ("kifach" in lower_text or "كيفاش" in lower_text or "n9ed" in lower_text or "n9dar" in lower_text or "symptom" in lower_text or "a3rad" in lower_text):
            return ("diabète", "Quels sont les symptômes du diabète et comment le diagnostiquer ?")

        # Rule 3: AVC Symptoms / Stroke
        if "jalta" in lower_text and ("dimagh" in lower_text or "mokh" in lower_text or "دماغ" in lower_text or "مخ" in lower_text or "avc" in lower_text):
            return ("AVC (Accident Vasculaire Cérébral)", "Quels sont les symptômes d'un accident vasculaire cérébral (AVC) et comment réagir ?")

        # Rule 4: Crisis / Heart Attack
        if "jalta" in lower_text and ("qalb" in lower_text or "قلب" in lower_text or "infarctus" in lower_text or "crise" in lower_text):
            return ("Infarctus du myocarde", "Quels sont les symptômes d'un infarctus du myocarde (crise cardiaque) ?")

        # Rule 5: Asthma in children
        if ("ddiq" in lower_text or "diiq" in lower_text or "الضيق" in lower_text or "asme" in lower_text or "asthme" in lower_text) and ("drari" in lower_text or "sghar" in lower_text or "ولد" in lower_text or "wld" in lower_text or "enfant" in lower_text):
            return ("asthme chez l'enfant", "Comment prendre en charge et gérer une crise d'asthme chez l'enfant ?")

        # Rule 6: Contraceptives / safe ones
        if ("mawanih" in lower_text or "mawani3" in lower_text or "موانع" in lower_text or "contracept" in lower_text) and ("haml" in lower_text or "حمل" in lower_text or "aamina" in lower_text or "safe" in lower_text):
            return ("contraception", "Quels sont les moyens de contraception sûrs et efficaces selon le guide du Ministère de la Santé ?")

        # General Category mappings (return category, no direct translation)
        if "skar" in lower_text or "sokar" in lower_text or "السكر" in lower_text or "diabet" in lower_text:
            return ("diabète", "")
        if "tansioun" in lower_text or "tansion" in lower_text or "tansiot" in lower_text or "طانسيو" in lower_text or "hta" in lower_text:
            return ("hypertension artérielle (HTA)", "")
        if "jalta" in lower_text or "avc" in lower_text:
            return ("complications cardiovasculaires", "")
        if "ddiq" in lower_text or "diiq" in lower_text or "asthme" in lower_text:
            return ("asthme", "")
        if "mawanih" in lower_text or "mawani3" in lower_text or "contracept" in lower_text:
            return ("contraception", "")
        if "haml" in lower_text or "grossesse" in lower_text:
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
            is_arabizi = any(w in user_query.lower() for w in darija_keywords) or any(c in user_query for c in ["3", "7", "9"])
            
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
                try:
                    res_trans = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": translation_prompt}],
                        options={"temperature": 0.0, "num_predict": 60},
                        keep_alive=OLLAMA_KEEP_ALIVE
                    )
                    translated = res_trans['message']['content'].strip()
                    translated = translated.replace('"', '').replace("'", "")
                    if translated:
                        query_fr = translated
                except Exception as e:
                    print(f"Few-shot translation error: {str(e)}")
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
                try:
                    res_trans = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": translation_prompt}],
                        options={"temperature": 0.0, "num_predict": 60},
                        keep_alive=OLLAMA_KEEP_ALIVE
                    )
                    translated = res_trans['message']['content'].strip()
                    translated = translated.replace('"', '').replace("'", "")
                    if translated:
                        query_fr = translated
                except Exception as e:
                    print(f"Few-shot translation error: {str(e)}")
                
        return {
            "is_darija": is_darija,
            "script_type": script_type,
            "query_fr": query_fr
        }
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

    def is_pure_greeting(self, text: str) -> bool:
        """
        Detects if a user message is a simple greeting (salam, hello, hi, etc.)
        and should bypass RAG retrieval.
        """
        # Clean text: lowercase, remove punctuation
        cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
        words = cleaned.split()
        
        if not words or len(words) > 3:
            return False
            
        greeting_keywords = {
            "salam", "salut", "hello", "hi", "hey", "bonjour", "bonsoir", "yo",
            "salamo3alikom", "salamo3alaykom", "salamo3likom", "salamoualikoum", "salamoualaykoum",
            "sbah", "lkhir", "lkheir", "msa", "msah", "labas", "cava", "ca", "va",
            "سلام", "السلام", "علكم", "عليكم", "صباح", "الخير", "مساء", "لاباس"
        }
        
        return all(word in greeting_keywords for word in words)

    def generate_response(self, original_query: str, query_analysis: dict, retrieved_chunks: list) -> str:
        """
        Generates the final response based on the retrieved French chunks using Ollama.
        Applies strict constraints to avoid hallucinations and responds in the correct language.
        """
        # Build context string
        context_str = ""
        for i, (chunk, score) in enumerate(retrieved_chunks):
            context_str += f"--- Source {i+1} (Fichier: {chunk['source']}, Page: {chunk.get('page')}) ---\n"
            context_str += f"{chunk['text']}\n\n"

        is_darija = query_analysis.get("is_darija", False)
        query_fr = query_analysis.get("query_fr", original_query)

        if is_darija:
            # Step 1: Generate response in French using query_fr (very accurate context extraction)
            system_prompt_fr = f"""Vous êtes un assistant documentaire officiel du Ministère de la Santé du Maroc.
Votre tâche est de répondre à la question de l'utilisateur en synthétisant UNIQUEMENT les informations du contexte officiel fourni ci-dessous.
Répondez de manière factuelle et directe en Français.

RÈGLES CRITIQUES :
1. Vous devez vous baser UNIQUEMENT sur le contexte fourni. N'inventez aucune information médicale.
2. Si le contexte ne contient pas l'information pour répondre, répondez exactement par: "NO_INFO"
3. Répondez directement en français simple et professionnel.

--- CONTEXTE OFFICIEL ---
{context_str}
"""
            try:
                res_fr_api = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt_fr},
                        {"role": "user", "content": query_fr}
                    ],
                    options={"temperature": 0.0, "num_predict": 400, "num_ctx": 2048},
                    keep_alive=OLLAMA_KEEP_ALIVE
                )
                res_fr = res_fr_api['message']['content'].strip()
                
                # Check for empty response or "NO_INFO"
                if "NO_INFO" in res_fr or len(res_fr) < 15:
                    return "ما عنديش هاد المعلومة فالمصادر الرسمية ديال وزارة الصحة."
                
                # Step 2: Translate the French response to clean Arabic/Darija script
                translation_to_ar_prompt = f"""ترجم النص الطبي التالي بدقة إلى اللغة العربية الفصحى البسيطة والواضحة. أجب فقط بالترجمة العربية ولا تضف أي نص آخر. ممنوع منعا تاما استخدام الأحرف الصينية أو أي لغة أخرى غير العربية.

النص بالفرنسية:
{res_fr}

الترجمة باللغة العربية:"""

                # Retry once if the small model leaks Chinese tokens, then strip as last resort
                res_ar = ""
                for _attempt in range(2):
                    res_ar_api = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": translation_to_ar_prompt}],
                        options={"temperature": 0.0, "num_predict": 600},
                        keep_alive=OLLAMA_KEEP_ALIVE
                    )
                    res_ar = res_ar_api['message']['content'].strip()
                    if not self._contains_cjk(res_ar):
                        break
                if self._contains_cjk(res_ar):
                    res_ar = self._strip_cjk(res_ar)
                
                # Clean up any residual prompt prefixes
                res_ar = res_ar.replace("الترجمة باللغة العربية:", "").replace("الترجمة باللغة العربية", "").strip()
                assistant_prefix = "الجواب الرسمي من وزارة الصحة المغربية هو: "
                return f"{assistant_prefix}{res_ar}"
            except Exception as e:
                print(f"Error in two-step Arabic response generation: {str(e)}")
                return "ما عنديش هاد المعلومة فالمصادر الرسمية ديال وزارة الصحة."
        else:
            # Classic French generation
            system_prompt = f"""Vous êtes un assistant documentaire officiel du Ministère de la Santé du Maroc.
Votre tâche est de répondre à la question de l'utilisateur en synthétisant UNIQUEMENT les informations du contexte officiel fourni ci-dessous.

RÈGLES CRITIQUES :
1. Vous devez vous baser UNIQUEMENT sur le contexte fourni. N'inventez aucune information médicale.
2. Si le contexte ne contient pas l'information pour répondre, répondez exactement par: "Désolé, je ne dispose pas d'informations officielles à ce sujet dans les documents du Ministère de la Santé."
3. Répondez en Français clair, professionnel et médicalement précis.

--- CONTEXTE OFFICIEL (en Français) ---
{context_str}
"""
            assistant_prefix = "Selon les documents officiels du Ministère de la Santé : "
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": original_query}
                    ],
                    options={"temperature": self.temperature, "num_predict": 400, "num_ctx": 2048},
                    keep_alive=OLLAMA_KEEP_ALIVE
                )
                generated_content = response['message']['content'].strip()
                return f"{assistant_prefix}{generated_content}"
            except Exception as e:
                print(f"Error during response generation locally: {str(e)}")
                return f"Error: Failed to generate response ({str(e)})"

    def answer_query(self, user_query: str, top_k: int = 5) -> dict:
        """
        Executes the entire local RAG pipeline:
        1. Checks if it is a pure greeting (bypasses RAG)
        2. Translation/Analysis of query
        3. Vector search in French document collection using local embeddings
        4. Response generation using local Ollama model
        Returns a dict with: 'response', 'retrieved_sources', 'query_fr', 'is_darija'
        """
        if self.vector_store.is_empty():
            return {
                "response": "La base de connaissances est vide. Indexation automatique en cours dans le thread d'arrière-plan...",
                "retrieved_sources": [],
                "query_fr": user_query,
                "is_darija": False
            }

        # 1. Check if it is a pure greeting
        if self.is_pure_greeting(user_query):
            analysis = self.detect_and_translate_query(user_query)
            is_arabic = (analysis.get("script_type") == "arabic")
            is_latin = (analysis.get("script_type") == "latin")
            
            # Select friendly system prompt based on language
            if is_arabic:
                sys_prompt = "أنت طبيبك (Tbibk)، مساعد طبي افتراضي رسمي لوزارة الصحة المغربية. رحب بالمستخدم بلطف بالدارجة المغربية (بالحروف العربية) واسأله كيف يمكنك مساعدته اليوم في صحته."
            elif is_latin:
                sys_prompt = "You are Tbibk, an official virtual medical assistant for the Moroccan Ministry of Health. Greet the user warmly in Moroccan Darija (Arabizi/latin script, e.g. using 3, 7, 9) and ask how you can help them with their health today."
            else:
                sys_prompt = "Vous êtes Tbibk, un assistant médical virtuel officiel du Ministère de la Santé du Maroc. Greet the user warmly in French and ask how you can help them with their health today."
                
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    options={"temperature": 0.5, "num_predict": 100},
                    keep_alive=OLLAMA_KEEP_ALIVE
                )
                response_text = response['message']['content'].strip()
            except Exception as e:
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
