import streamlit as st
from ask import ask_question

st.set_page_config(
    page_title="RAG simple",
    page_icon="📄",
)

st.title("RAG simple — Questions Maths :")

st.write(
    "Pose une question. Le modèle répondra à partir des documents indexés."
)

question = st.text_input("Votre question")

if st.button("Répondre"):
    if not question.strip():
        st.warning("Merci d’écrire une question simple.")
    else:
        with st.spinner("Recherche dans les documents..."):
            answer = ask_question(question)

        st.subheader("Réponse")
        st.write(answer)
