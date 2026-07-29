# 🚀 Guide de Déploiement - Tbibk (طبيبك)

Ce document explique comment déployer l'application **Tbibk** en production.

---

## Méthode 1 : Déploiement Gratuit sur Streamlit Community Cloud (Recommandé pour tester)

### Étape 1 : Publier votre code sur GitHub
1. Créez un nouveau dépôt sur GitHub (ex: `tbibk-medical-rag`).
2. Publiez votre code sur GitHub :
   ```bash
   git add .
   git commit -m "Préparation pour le déploiement Streamlit"
   git push origin main
   ```

### Étape 2 : Connecter le dépôt à Streamlit Cloud
1. Rendez-vous sur **[share.streamlit.io](https://share.streamlit.io)** et connectez votre compte GitHub.
2. Cliquez sur **"New app"**.
3. Sélectionnez votre dépôt (`tbibk-medical-rag`), la branche (`main`) et le fichier principal (`app.py`).
4. Cliquez sur **"Deploy!"**.

---

## Méthode 2 : Déploiement 100% Local / Cloud VPS avec Docker (Recommandé pour Production Privée)

Cette méthode déploie **Streamlit + Ollama (Qwen 2.5 1.5B)** ensemble dans des conteneurs Docker isolés sur n'importe quel serveur VPS (Hetzner, DigitalOcean, AWS, OVH).

### Étape 1 : Prérequis sur le serveur VPS
Installez Docker et Docker Compose sur le serveur :
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
```

### Étape 2 : Cloner et Lancer l'Application
```bash
git clone <URL_DE_VOTRE_REPO_GITHUB>
cd "medical RAG chatbot"

# Lancement des conteneurs (Streamlit + Ollama)
docker-compose up -d --build

# Télécharger le modèle Qwen dans le conteneur Ollama
docker exec -it tbibk_ollama ollama pull qwen2.5:1.5b
```

Votre application est désormais accessible en ligne sur le port `8501` de votre serveur VPS (ex: `http://IP_VOTRE_SERVEUR:8501`).

---

## Méthode 3 : Déploiement sur Hugging Face Spaces (Gratuit)

1. Créez un nouveau Space sur [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choisissez le SDK **Streamlit** ou **Docker**.
3. Poussez les fichiers du projet sur votre Space Hugging Face.
