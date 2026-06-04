import sys
from pathlib import Path

import streamlit as st  # Framework pour créer des interfaces web en pur Python

# Ajoute le dossier src/ au chemin Python pour pouvoir importer ask.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ask_question  # Fonction principale du RAG (recherche + génération)

st.title("RAG Aquila")  # Titre affiché en haut de la page web

# Champ de saisie texte pour la question de l'utilisateur
question = st.text_input("Votre question :")

# Le bouton "Envoyer" déclenche la recherche uniquement si une question a été saisie
if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):  # Affiche un indicateur de chargement pendant le traitement
        reponse = ask_question(question)       # Appel au pipeline RAG complet
    st.write(reponse)  # Affiche la réponse générée par le LLM
