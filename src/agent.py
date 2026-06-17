import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from ask import retrieve, _rerank, llm, _invoke_with_retry, _maybe_restart_ollama, PROMPT_PATH, BASE_DIR, K_FINAL

# Détecte les marqueurs de coupure de page produits par pymupdf4llm
# Exemples : "i— PAGE **19** SUR 78 >", "PAGE 9 SUR 78"
TRUNCATION_RE = re.compile(r'(?:i—\s*)?PAGE\s+\*{0,2}\d+\*{0,2}\s+SUR\s+\d+', re.IGNORECASE)

# Dossier des documents source — on liste son contenu dynamiquement (pas de nom d'établissement
# en dur) pour que l'agent reste valable si on ajoute/retire des brochures plus tard.
DOCUMENTS_DIR = BASE_DIR / "documents"

# Nombre max de reformulations de requête avant de générer quand même avec ce qu'on a.
# 1 = exactement 2 retrievals possibles : l'initial + 1 reformulation ciblée.
MAX_ATTEMPTS = 1

SOURCE_AND_DIFFICULTY_PROMPT = """Voici la liste des documents disponibles dans la base : {sources}

Voici la question posée : {question}

Réponds en DEUX lignes exactement, dans cet ordre :

SOURCES: [le(s) nom(s) de fichier(s) exact(s) séparés par une virgule, ou le mot TOUS]
DIFFICULTE: [1, 2 ou 3]

Règles pour SOURCES :
- Si la question mentionne explicitement un établissement ou un diplôme propre à un établissement, indique le fichier correspondant.
- Si la question compare plusieurs établissements ou ne mentionne aucun établissement précis, écris TOUS.

Règles pour DIFFICULTE :
- 1 = factuelle simple : cherche une valeur unique (nom, date, nombre, durée, liste courte) dans un seul document. Réponse en une phrase.
- 2 = synthèse : demande une explication, un fonctionnement, ou plusieurs informations d'un même document.
- 3 = complexe : comparaison entre documents, raisonnement croisé, ou question ouverte sans réponse directe.

Exemples :
Question : "Qui dirige le DMA en 2024-2025 ?" → SOURCES: ENS.pdf\nDIFFICULTE: 1
Question : "Comment fonctionne le système de tutorat ?" → SOURCES: ENS.pdf\nDIFFICULTE: 2
Question : "Quelles sont les différences entre les stages ENS et Sorbonne ?" → SOURCES: TOUS\nDIFFICULTE: 3

Ne réponds rien d'autre que ces deux lignes."""

GRADE_PROMPT = """Voici une question et des extraits de documents récupérés pour y répondre.

Question :
{question}

Extraits récupérés :
{context}

Ces extraits contiennent-ils l'information nécessaire pour répondre correctement à la question ?
- Si oui, réponds uniquement : OUI
- Si non, réponds : NON — [explique en 1 phrase ce qui manque précisément dans les extraits]
- Si un extrait est marqué [⚠ TRONQUÉ], considère que la liste ou la phrase est incomplète et réponds NON en précisant que la suite manque.

Exemples de réponse NON :
NON — les extraits mentionnent le stage mais n'indiquent pas sa durée minimale
NON — aucun extrait ne précise les conditions géographiques requises
NON — l'extrait est tronqué, les cours du 2ème semestre et le stage sont absents"""

REWRITE_PROMPT = """La recherche suivante n'a pas permis de retrouver une information suffisante pour répondre à la question.

Question originale : {question}
Requête de recherche utilisée jusqu'ici : {query}
Ce qui manque selon l'analyse des extraits : {verdict}

Propose une requête de recherche ciblée sur ce qui manque (synonymes, mots-clés différents, reformulation).

Réponds uniquement par la nouvelle requête, sans explication ni guillemets."""


class AgentState(TypedDict):
    question: str               # la question originale, ne change jamais
    current_query: str          # la requête de recherche ACTUELLE (peut être reformulée en boucle)
    sources: list[str] | None   # source(s) identifiée(s) par identify_sources, ou None = chercher partout
    difficulty: int             # 1=factuel, 2=synthèse, 3=complexe — peut être remonté dynamiquement
    initial_difficulty: int     # difficulté classifiée par identify_sources — pour les logs
    docs: list[Document]        # pool cumulatif de tous les chunks récupérés (tous retrievals confondus)
    sufficient: bool            # verdict du dernier passage dans grade_documents
    grade_verdict: str          # verdict complet du grade (ex: "NON — durée du stage absente")
    attempts: int               # nombre de retrievals déjà effectués — sert à plafonner la boucle
    answer: str                 # réponse finale produite par generate_node


def _available_sources() -> list[str]:
    """Liste les fichiers du dossier documents/, dans le même ordre que load_documents() dans ingest.py."""
    return sorted(p.name for p in DOCUMENTS_DIR.iterdir() if not p.name.startswith("."))


def identify_sources(state: AgentState) -> dict:
    """
    Combine en un seul appel LLM l'identification de la/les source(s) ET la classification
    de difficulté (1=factuel, 2=synthèse, 3=complexe). Zéro coût supplémentaire par rapport
    à l'ancienne version — la difficulté est parsée depuis la même réponse.
    Fallback : sources=None (cherche partout) et difficulty=2 si le LLM ne suit pas le format.
    """
    available = _available_sources()
    prompt = SOURCE_AND_DIFFICULTY_PROMPT.format(sources=", ".join(available), question=state["question"])
    raw = _invoke_with_retry(prompt).strip()

    # Parsing ligne par ligne — robuste aux sauts de ligne et aux espaces superflus
    identified = None
    difficulty = 2  # fallback : pipeline standard si le LLM dévie du format
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("SOURCES:"):
            val = line.split(":", 1)[1].strip()
            if "tous" not in val.lower():
                matched = [name for name in available if name.lower() in val.lower()]
                identified = matched or None
        elif line.upper().startswith("DIFFICULTE:"):
            val = line.split(":", 1)[1].strip()
            if val in ("1", "2", "3"):
                difficulty = int(val)

    return {"sources": identified, "difficulty": difficulty, "initial_difficulty": difficulty, "current_query": state["question"], "attempts": 0}


def retrieve_node(state: AgentState) -> dict:
    """Lance le retrieval et construit le pool final selon le numéro de tentative.
    HyDE est désactivé pour les questions de difficulté 1 (économise un appel LLM)."""
    use_hyde = state["difficulty"] > 1
    new_docs = retrieve(state["current_query"], sources=state["sources"], verbose=False, use_hyde=use_hyde)
    # Déduplique par contenu : évite d'envoyer deux fois le même chunk au LLM
    existing_contents = {d.page_content for d in state["docs"]}
    new_docs_deduped = [d for d in new_docs if d.page_content not in existing_contents]

    if state["attempts"] == 0:
        # 1er retrieval : re-rank libre sur tous les chunks récupérés
        merged = state["docs"] + new_docs_deduped
        final = _rerank(state["question"], merged) if len(merged) > K_FINAL else merged
    else:
        # 2ème retrieval — Option D : 3 slots pour l'ancien pool + 2 slots réservés aux nouveaux chunks.
        # Garantit que les chunks ciblant l'info manquante arrivent au LLM même s'ils auraient
        # perdu face aux chunks généraux du 1er retrieval dans un re-rank global.
        old_top = state["docs"][:K_FINAL - 2]  # déjà triés par re-rank sur la question originale
        new_top = _rerank(state["current_query"], new_docs_deduped, n=2) if new_docs_deduped else []
        seen = {d.page_content for d in old_top}
        final = old_top + [d for d in new_top if d.page_content not in seen]

    return {"docs": final, "attempts": state["attempts"] + 1}


def _has_truncation(docs: list[Document]) -> bool:
    """Détecte si un chunk se termine par un marqueur de coupure de page pymupdf4llm."""
    return any(TRUNCATION_RE.search(doc.page_content[-150:]) for doc in docs)


def _annotate_context(docs: list[Document]) -> str:
    """Construit le contexte pour grade_documents en signalant les chunks tronqués."""
    parts = []
    for doc in docs:
        content = doc.page_content
        if TRUNCATION_RE.search(content[-150:]):
            content += "\n[⚠ TRONQUÉ — la liste ou la phrase continue sur la page suivante]"
        parts.append(content)
    return "\n\n---\n\n".join(parts)


def grade_documents(state: AgentState) -> dict:
    """Demande au LLM si les chunks accumulés suffisent, et ce qui manque précisément si non.
    Les chunks tronqués sont annotés pour que le LLM détecte les listes incomplètes."""
    if not state["docs"]:
        return {"sufficient": False, "grade_verdict": "NON — aucun extrait récupéré"}
    context = _annotate_context(state["docs"])
    prompt = GRADE_PROMPT.format(question=state["question"], context=context)
    raw = _invoke_with_retry(prompt).strip()
    sufficient = raw.lower().startswith("oui")
    return {"sufficient": sufficient, "grade_verdict": raw}


def rewrite_query(state: AgentState) -> dict:
    """Reformule la requête en ciblant précisément ce qui manque selon le verdict du grade."""
    prompt = REWRITE_PROMPT.format(
        question=state["question"],
        query=state["current_query"],
        verdict=state["grade_verdict"],   # transmet ce qui manque pour une reformulation ciblée
    )
    new_query = _invoke_with_retry(prompt).strip()
    return {"current_query": new_query}


def upgrade_difficulty(state: AgentState) -> dict:
    """Remonte la difficulté à 3 pour débloquer la reformulation.
    Appelé quand un retrieval de niveau 1 (troncature détectée) ou 2 (grade NON) s'avère insuffisant."""
    return {"difficulty": 3}


def generate_node(state: AgentState) -> dict:
    """Génère la réponse finale à partir du pool cumulatif de tous les chunks récupérés."""
    if not state["docs"]:
        return {"answer": "Je ne trouve pas cette information dans les documents fournis."}
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in state["docs"]
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=state["question"], context=context)
    return {"answer": _invoke_with_retry(prompt)}


def _route_after_retrieve(state: AgentState) -> str:
    """
    Difficulté 1 : vérifie la troncature (sans appel LLM).
      → troncature détectée : upgrade_difficulty pour débloquer la reformulation
      → pas de troncature : generate directement
    Difficulté 2/3 : grade_documents.
    """
    if state["difficulty"] == 1:
        if _has_truncation(state["docs"]):
            return "upgrade_difficulty"
        return "generate"
    return "grade_documents"


def _route_after_grading(state: AgentState) -> str:
    """
    Suffisant ou quota atteint : generate.
    Insuffisant + difficulté < 3 : upgrade_difficulty pour débloquer la reformulation.
    Insuffisant + difficulté 3 + tentatives restantes : rewrite_query.
    """
    if state["sufficient"] or state["attempts"] > MAX_ATTEMPTS:
        return "generate"
    if state["difficulty"] < 3:
        return "upgrade_difficulty"
    return "rewrite_query"


def _build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("identify_sources", identify_sources)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("upgrade_difficulty", upgrade_difficulty)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "identify_sources")
    graph.add_edge("identify_sources", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "grade_documents": "grade_documents", "upgrade_difficulty": "upgrade_difficulty"},
    )
    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {"generate": "generate", "upgrade_difficulty": "upgrade_difficulty", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("upgrade_difficulty", "rewrite_query")
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
    _maybe_restart_ollama(verbose=verbose)
    final_state = agent.invoke({
        "question": question,
        "current_query": question,
        "sources": None,
        "difficulty": 2,
        "initial_difficulty": 2,  # écrasé par identify_sources
        "docs": [],
        "sufficient": False,
        "grade_verdict": "",
        "attempts": 0,
        "answer": "",
    })

    if verbose:
        diff_labels = {1: "factuel", 2: "synthèse", 3: "complexe"}
        init_d = final_state["initial_difficulty"]
        final_d = final_state["difficulty"]
        upgrade_str = f" → remontée à 3 (pipeline complet)" if final_d > init_d else ""
        print(f"[Agent] Difficulté : {init_d} — {diff_labels.get(init_d, '?')}{upgrade_str}")
        print(f"[Agent] Source(s) ciblée(s) : {', '.join(final_state['sources']) if final_state['sources'] else 'toutes (pas de filtre)'}")
        print(f"[Agent] Tentative(s) de retrieval : {final_state['attempts']}")
        if final_d > 1 or final_state["attempts"] > 1:
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
