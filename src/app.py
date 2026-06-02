"""
app.py

Interface Streamlit très simple pour poser des questions aux documents.
Lancer avec :
streamlit run src/app.py
"""

import streamlit as st
from ask import ask_question

st.set_page_config(
    page_title="RAG simple",
    page_icon="📄",
)

st.title("RAG simple — Questions sur documents")

st.write(
    "Pose une question. Le modèle répondra à partir des documents indexés."
)

question = st.text_input("Votre question")

if st.button("Répondre"):
    if not question.strip():
        st.warning("Merci d’écrire une question.")
    else:
        with st.spinner("Recherche dans les documents..."):
            answer = ask_question(question)

        st.subheader("Réponse")
        st.write(answer)
