import sys
from pathlib import Path

import streamlit as st  # framework pour créer une interface web en Python

# Ajoute src/ au chemin pour pouvoir importer ask.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ask_question  # pipeline RAG complet (recherche + génération)

st.title("RAG Aquila")

question = st.text_input("Votre question :")

if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):
        reponse, _ = ask_question(question)
    st.write(reponse)
