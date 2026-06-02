from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gemma:2b"

llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096)


def ask_question(question: str) -> str:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
    docs = vector_db.as_retriever(search_kwargs={"k": 4}).invoke(question)

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "source inconnue")
        context_parts.append(f"Source : {source}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.format(question=question, context=context)

    return llm.invoke(prompt)


if __name__ == "__main__":
    question = input("Pose ta question : ")
    print("\nRéponse :")
    print(ask_question(question))
