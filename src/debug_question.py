"""
Script de debug ponctuel : exécute le pipeline de retrieval (verbose) puis le pipeline
agentique complet sur une question donnée, pour analyser pourquoi certains chunks
pertinents ne sont pas récupérés.

Usage : python debug_question.py
"""

from ask import retrieve
from agent import ask_question_agentic

QUESTION = (
    "Quel est le programme provisoire détaillé du cours d'Algèbre 1 (L3 S1) de l'ENS DMA, "
    "et qui sont ses enseignants ?"
)

print("=" * 80)
print("ÉTAPE 1 : retrieve() seul, verbose, filtré sur ENS.pdf")
print("=" * 80)
docs = retrieve(QUESTION, sources=["ENS.pdf"], verbose=True)

print("\n" + "=" * 80)
print("ÉTAPE 2 : pipeline agentique complet (identify_sources -> retrieve -> grade -> ...)")
print("=" * 80)
answer, final_docs = ask_question_agentic(QUESTION, verbose=True)

print("\n" + "=" * 80)
print("RÉPONSE FINALE")
print("=" * 80)
print(answer)

print("\n" + "=" * 80)
print(f"CHUNKS FINAUX UTILISÉS ({len(final_docs)})")
print("=" * 80)
for i, doc in enumerate(final_docs):
    print(f"\n--- Chunk #{i+1} — {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} ---")
    print(doc.page_content)
