from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from ask import retrieve, llm, PROMPT_PATH, BASE_DIR

# Dossier des documents source — on liste son contenu dynamiquement (pas de nom d'établissement
# en dur) pour que l'agent reste valable si on ajoute/retire des brochures plus tard.
DOCUMENTS_DIR = BASE_DIR / "documents"

# Nombre max de reformulations de requête avant de générer quand même avec ce qu'on a.
# 1 = au pire 2 tentatives de retrieval (~2x plus de temps que le RAG classique en cas de boucle).
MAX_ATTEMPTS = 1

SOURCE_PROMPT = """Voici la liste des documents disponibles dans la base : {sources}

Voici la question posée : {question}

Quel(s) document(s) de cette liste sont nécessaires pour répondre à cette question ?
- Si la question mentionne explicitement un établissement (ex: ENS, Sorbonne Université) ou un diplôme propre à un établissement (ex: DENS), choisis le document de cet établissement.
- Si la question compare plusieurs établissements, ou ne mentionne aucun établissement et pourrait concerner n'importe lequel, écris "TOUS".

Exemples :
Question : "Quelle est la durée totale de la formation du Diplôme de l'ENS ès Mathématiques (DENS) ?" → ENS.pdf
Question : "Combien d'ECTS faut-il valider en Master 2 à Sorbonne Université ?" → SORBONNE.pdf
Question : "Quelles sont les différences entre les stages de l'ENS et de Sorbonne Université ?" → TOUS

Réponds uniquement par le(s) nom(s) de fichier(s) exact(s) séparés par une virgule, ou par "TOUS"."""

GRADE_PROMPT = """Voici une question et des extraits de documents récupérés pour y répondre.

Question :
{question}

Extraits récupérés :
{context}

Ces extraits contiennent-ils l'information nécessaire pour répondre correctement à la question ?
Réponds uniquement par OUI ou NON."""

REWRITE_PROMPT = """La recherche suivante n'a pas permis de retrouver une information suffisante pour répondre à la question.

Question originale : {question}
Requête de recherche utilisée jusqu'ici : {query}

Propose une autre formulation de requête de recherche (synonymes, mots-clés différents, reformulation),
qui pourrait mieux retrouver l'information dans les documents.

Réponds uniquement par la nouvelle requête, sans explication ni guillemets."""


class AgentState(TypedDict):
    question: str               # la question originale, ne change jamais
    current_query: str          # la requête de recherche ACTUELLE (peut être reformulée en boucle)
    sources: list[str] | None   # source(s) identifiée(s) par identify_sources, ou None = chercher partout
    docs: list[Document]        # derniers chunks récupérés par retrieve_node
    first_docs: list[Document]  # chunks du tout premier retrieval (avant toute reformulation)
    sufficient: bool            # verdict du dernier passage dans grade_documents
    attempts: int               # nombre de retrievals déjà effectués — sert à plafonner la boucle
    answer: str                 # réponse finale produite par generate_node


def _available_sources() -> list[str]:
    """Liste les fichiers du dossier documents/, dans le même ordre que load_documents() dans ingest.py."""
    return sorted(p.name for p in DOCUMENTS_DIR.iterdir() if not p.name.startswith("."))


def identify_sources(state: AgentState) -> dict:
    """
    Demande au LLM quel(s) document(s) sont concernés par la question, en lui donnant
    la liste réelle des fichiers présents dans documents/ (découverte dynamique — aucun
    nom d'établissement n'est codé en dur, donc ça reste valable si on ajoute des brochures).
    Si la réponse ne correspond à aucun nom connu, ou si elle dit "TOUS", on ne filtre pas
    (sources=None) : retrieve() cherchera alors dans toute la base.
    """
    available = _available_sources()
    prompt = SOURCE_PROMPT.format(sources=", ".join(available), question=state["question"])
    raw = llm.invoke(prompt).strip()

    if "tous" in raw.lower():
        identified = None
    else:
        # Ne garde que les noms qui correspondent réellement à un fichier existant
        # (le LLM peut halluciner un nom approchant — on ne lui fait pas confiance aveuglément)
        matched = [name for name in available if name.lower() in raw.lower()]
        identified = matched or None

    return {"sources": identified, "current_query": state["question"], "attempts": 0}


def retrieve_node(state: AgentState) -> dict:
    """Lance le pipeline de retrieval existant (sémantique + BM25 + RRF + reranking), filtré sur les sources identifiées."""
    docs = retrieve(state["current_query"], sources=state["sources"], verbose=False)
    update = {"docs": docs, "attempts": state["attempts"] + 1}
    if state["attempts"] == 0:  # premier retrieval (requête originale, avant toute reformulation)
        update["first_docs"] = docs
    return update


def grade_documents(state: AgentState) -> dict:
    """Demande au LLM si les chunks récupérés suffisent pour répondre à la question d'origine."""
    if not state["docs"]:
        return {"sufficient": False}
    context = "\n\n---\n\n".join(doc.page_content for doc in state["docs"])
    prompt = GRADE_PROMPT.format(question=state["question"], context=context)
    raw = llm.invoke(prompt).strip().lower()
    return {"sufficient": raw.startswith("oui")}


def rewrite_query(state: AgentState) -> dict:
    """Demande au LLM de reformuler la requête de recherche pour tenter de mieux retrouver l'information."""
    prompt = REWRITE_PROMPT.format(question=state["question"], query=state["current_query"])
    new_query = llm.invoke(prompt).strip()
    return {"current_query": new_query}


def generate_node(state: AgentState) -> dict:
    """
    Génère la réponse finale à partir des chunks récupérés (même logique que ask_question).
    Si la reformulation n'a pas permis d'obtenir un verdict "suffisant", on revient aux
    chunks du tout premier retrieval plutôt que d'utiliser ceux de la requête reformulée :
    en pratique la reformulation dérive parfois vers une requête moins pertinente
    (mauvaise source, voire 0 chunk), alors que la requête d'origine retrouvait déjà
    les bons passages.
    """
    docs = state["docs"] if state["sufficient"] else (state["first_docs"] or state["docs"])
    if not docs:
        return {"answer": "Je ne trouve pas cette information dans les documents fournis."}
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in docs
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=state["question"], context=context)
    return {"answer": llm.invoke(prompt)}


def _route_after_grading(state: AgentState) -> str:
    """
    Conditionne la suite du graphe après grade_documents :
    - chunks suffisants → on génère directement
    - chunks insuffisants mais il reste des tentatives → on reformule et on recherche à nouveau
    - chunks insuffisants et plus de tentatives → on génère quand même avec ce qu'on a
      (mieux qu'une boucle infinie ou qu'un échec sec)
    """
    if state["sufficient"] or state["attempts"] > MAX_ATTEMPTS:
        return "generate"
    return "rewrite_query"


def _build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("identify_sources", identify_sources)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "identify_sources")
    graph.add_edge("identify_sources", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# Compilé une seule fois au chargement du module (comme llm/reranker dans ask.py)
agent = _build_agent()


def ask_question_agentic(question: str, verbose: bool = True) -> tuple[str, list[Document]]:
    """
    Pipeline RAG agentique complet : identification de la/les source(s) → retrieval →
    évaluation des chunks → (reformulation + nouveau retrieval si besoin, dans la limite
    de MAX_ATTEMPTS) → génération.
    Retourne un tuple (réponse_texte, chunks_utilisés), comme ask_question, pour rester
    interchangeable avec evaluate.py / app.py.
    """
    final_state = agent.invoke({
        "question": question,
        "current_query": question,
        "sources": None,
        "docs": [],
        "first_docs": [],
        "sufficient": False,
        "attempts": 0,
        "answer": "",
    })

    if verbose:
        print(f"[Agent] Source(s) ciblée(s) : {', '.join(final_state['sources']) if final_state['sources'] else 'toutes (pas de filtre)'}")
        print(f"[Agent] Tentative(s) de retrieval : {final_state['attempts']}")
        print(f"[Agent] Chunks jugés suffisants : {'oui' if final_state['sufficient'] else 'non'}")

    return final_state["answer"], final_state["docs"]


if __name__ == "__main__":
    # Mode terminal : boucle interactive, comme dans ask.py, mais avec le pipeline agentique
    print("=== RAG agentique — Mode terminal ===")
    print("Tapez votre question et appuyez sur Entrée. Ctrl+C pour quitter.")
    print("Attention : chaque question peut prendre plusieurs minutes (plusieurs appels LLM en chaîne).\n")
    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue
            print("Recherche en cours (peut prendre plusieurs minutes)...")
            answer, _ = ask_question_agentic(question)
            print(f"\nRéponse :\n{answer}")
            print("\n" + "-" * 60 + "\n")
        except ConnectionError:
            print("\n[Erreur] Ollama n'est pas accessible. Vérifiez qu'il tourne (`ollama serve`).\n")
        except KeyboardInterrupt:
            print("\nAu revoir.")
            break
        except Exception as e:
            print(f"\n[Erreur inattendue] {type(e).__name__}: {e}\n")
