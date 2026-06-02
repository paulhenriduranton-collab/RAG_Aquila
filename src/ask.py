from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gemma:2b"
SCORE_THRESHOLD = 0.3

llm = OllamaLLM(model=GEN_MODEL, num_ctx=8192, temperature=0)


def ask_question(question: str) -> str:
    print(f"\n[DB] Dossier vector_db : {VECTOR_DB_DIR}")
    print(f"[DB] Existe : {VECTOR_DB_DIR.exists()}")
    if VECTOR_DB_DIR.exists():
        fichiers = list(VECTOR_DB_DIR.iterdir())
        print(f"[DB] Fichiers : {[f.name for f in fichiers]}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )

    total = vector_db._collection.count()
    print(f"[DB] Nombre de chunks dans la base : {total}")

    raw = vector_db.similarity_search_with_relevance_scores(question, k=5)
    print(f"[DB] Résultats bruts retournés par Chroma : {len(raw)}")

    print("\n[Chunks récupérés]")
    for doc, score in raw:
        source = doc.metadata.get("source", "?")
        status = "OK" if score >= SCORE_THRESHOLD else "--"
        print(f"  [{status}] score={score:.3f}  {source}")

    docs = [doc for doc, score in raw if score >= SCORE_THRESHOLD]

    if not docs:
        print(f"[Aucun chunk ne dépasse le seuil {SCORE_THRESHOLD}]\n")
        return "Je ne trouve pas cette information dans les documents fournis."

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "source inconnue")
        context_parts.append(f"Source : {source}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.format(question=question, context=context)

    return llm.invoke(prompt)


if __name__ == "__main__":
    print("=== RAG — Mode terminal ===")
    print("Tapez votre question et appuyez sur Entrée. Ctrl+C pour quitter.\n")
    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue
            print("Recherche en cours...")
            answer = ask_question(question)
            print(f"\nRéponse :\n{answer}")
            print("\n" + "-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nAu revoir.")
            break
