import sys
from pathlib import Path

import streamlit as st  # framework pour créer une interface web en Python

# Ajoute src/ au chemin pour pouvoir importer ask.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ask_question  # pipeline RAG complet (recherche + génération)

st.title("RAG Aquila")

# Champ de saisie texte pour la question de l'utilisateur
question = st.text_input("Votre question :")

# Déclenche la recherche uniquement quand l'utilisateur clique sur Envoyer ET qu'une question est saisie
if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):  # affiche un indicateur de chargement pendant le traitement
        reponse, _ = ask_question(question, verbose=False)  # verbose=False : pas de logs dans l'interface
    st.write(reponse)  # affiche la réponse générée par le LLM
