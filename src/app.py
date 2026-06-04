import sys
from pathlib import Path

import streamlit as st

# Nécessaire pour importer ask.py depuis le même dossier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ask_question

st.title("RAG Aquila")

question = st.text_input("Votre question :")

if st.button("Envoyer") and question:
    with st.spinner("Recherche en cours..."):
        reponse = ask_question(question)
    st.write(reponse)
