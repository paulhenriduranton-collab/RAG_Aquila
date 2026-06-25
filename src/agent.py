import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from ask import retrieve, _rerank, llm, _invoke_with_retry, _maybe_restart_ollama, PROMPT_PATH, BASE_DIR, K_FINAL


# Dossier des documents source — on liste son contenu dynamiquement (pas de nom d'établissement
# en dur) pour que l'agent reste valable si on ajoute/retire des brochures plus tard.
DOCUMENTS_DIR = BASE_DIR / "documents"

# Nombre max de reformulations de requête avant de générer quand même avec ce qu'on a.
# 1 = exactement 2 retrievals possibles : l'initial + 1 reformulation ciblée.
MAX_ATTEMPTS = 1

SOURCE_AND_DIFFICULTY_PROMPT = """Voici la liste des documents disponibles : {sources}

Question : {question}

Réponds dans ce format exact, sans rien d'autre :

SOURCES: [nom(s) de fichier(s) ou TOUS]
DIFFICULTE: [1, 2 ou 3]
SOUS-REQUETE-1: [uniquement si DIFFICULTE est 3]
SOUS-REQUETE-2: [uniquement si DIFFICULTE est 3 et que la question compare deux entités distinctes]

Règles SOURCES :
- Établissement ou diplôme explicite → fichier correspondant
- Comparaison ou aucun établissement précis → TOUS

Règles DIFFICULTE :
- 1 = factuelle : valeur unique (nom, date, nombre, liste courte), réponse en une phrase
- 2 = synthèse : explication, fonctionnement, plusieurs infos d'un même document
- 3 = comparaison entre documents ou sections, raisonnement croisé, question sans réponse directe

Règles SOUS-REQUETE (uniquement si DIFFICULTE: 3) :
- Décompose en 2 sous-requêtes indépendantes, chacune ciblant une seule entité ou section
- Si la question ne compare qu'une seule chose, écris une seule SOUS-REQUETE-1

Exemples :
Question : "Qui dirige le DMA ?" → SOURCES: ENS.pdf\nDIFFICULTE: 1
Question : "Comment fonctionne le tutorat ?" → SOURCES: ENS.pdf\nDIFFICULTE: 2
Question : "Compare les ECTS L3 et M1 ENS" → SOURCES: ENS.pdf\nDIFFICULTE: 3\nSOUS-REQUETE-1: ECTS cours L3 première année ENS tableau\nSOUS-REQUETE-2: ECTS cours fondamentaux M1 ENS tableau"""

GRADE_PROMPT = """Voici une question et des extraits de documents récupérés pour y répondre.

Question :
{question}

Extraits récupérés :
{context}

Ces extraits contiennent-ils l'information nécessaire pour répondre correctement à la question ?

ATTENTION — Avant de répondre OUI, vérifie :
- Que l'information trouvée correspond bien à l'entité PRÉCISE de la question (bon cours, bon programme, bon enseignant, bonne année, etc.), et pas à une entité voisine présente dans le même extrait.
- Qu'un extrait peut contenir des informations de PLUSIEURS sections ou cours différents mélangés. Ne pas attribuer une information (horaires, ECTS, enseignant...) à une entité simplement parce qu'elles apparaissent dans le même extrait.

- Si oui, réponds uniquement : OUI
- Si non, réponds : NON — [explique en 1 phrase ce qui manque précisément dans les extraits]

Exemples de réponse NON :
NON — les extraits mentionnent le stage mais n'indiquent pas sa durée minimale
NON — aucun extrait ne précise les conditions géographiques requises
NON — les horaires présents dans l'extrait appartiennent à un autre cours, pas à celui demandé"""

REWRITE_PROMPT = """La recherche suivante n'a pas permis de retrouver une information suffisante pour répondre à la question.

Question originale : {question}
Requête de recherche utilisée jusqu'ici : {query}
Ce qui manque selon l'analyse des extraits : {verdict}

Génère une nouvelle requête de recherche en respectant ces règles :
1. CONSERVE OBLIGATOIREMENT les identifiants de contexte de la question originale : nom d'établissement (ENS, Sorbonne…), nom de programme/parcours, année (L3, M1, M2…), nom de cours ou d'enseignant s'il y en a.
2. AJOUTE des mots-clés ciblant précisément ce qui manque (synonymes, termes techniques alternatifs, noms de sections du document).
3. Produis une liste de mots-clés séparés par des espaces, PAS une question ni une phrase complète.

Réponds uniquement par la nouvelle requête, sans explication ni guillemets."""


class AgentState(TypedDict):
    question: str               # la question originale, ne change jamais
    current_query: str          # la requête de recherche ACTUELLE (peut être reformulée en boucle)
    sources: list[str] | None   # source(s) identifiée(s) par identify_sources, ou None = chercher partout
    difficulty: int             # 1=factuel, 2=synthèse, 3=complexe — peut être remonté dynamiquement
    initial_difficulty: int     # difficulté classifiée par identify_sources — pour les logs
    sub_queries: list[str]       # sous-requêtes issues de decompose_query (vide si difficulté < 3)
    docs: list[Document]        # pool cumulatif de tous les chunks récupérés (tous retrievals confondus)
    sufficient: bool            # verdict du dernier passage dans grade_documents
    grade_verdict: str          # verdict complet du grade (ex: "NON — durée du stage absente")
    attempts: int               # nombre de retrievals déjà effectués — sert à plafonner la boucle
    answer: str                 # réponse finale produite par generate_node
    pre_rerank_docs: list[Document]      # chunks juste avant le re-ranking du 1er retrieval — debug/éval (③)
    docs_before_rewrite: list[Document]  # pool de chunks au moment du grading initial, avant rewrite — debug/éval (⑤)
    post_rewrite_docs: list[Document]    # chunks récupérés spécifiquement par le 2ème retrieval (post-rewrite) — debug
    pre_dedup_docs: list[Document]       # chunks après fusion RRF, avant déduplication — debug/éval (⑦ dédup)
    semantic_docs: list[Document]        # résultats sémantiques bruts (avant fusion) — debug/éval (⑧ RRF)
    bm25_docs: list[Document]            # résultats BM25 bruts (avant fusion) — debug/éval (⑧ RRF)
    dedup_removed_docs: list[Document]   # chunks écartés par la dédup — debug/éval (⑦ dédup)
    dedup_kept_docs: list[Document]      # chunk conservé qui a remplacé chaque écarté (même index) — debug/éval (⑦ dédup)
    grading_before_rewrite: dict         # snapshot {sufficient, verdict} du grading qui a déclenché le rewrite — debug/éval (④)


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
    difficulty = 2  # fallback
    sub_queries = []
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
        elif line.upper().startswith("SOUS-REQUETE-"):
            val = line.split(":", 1)[1].strip() if ":" in line else ""
            if val:
                sub_queries.append(val)

    # Si le LLM a listé TOUS les fichiers disponibles, c'est équivalent à TOUS → pas de filtre
    if identified and set(identified) == set(available):
        identified = None

    # Pour difficulté 3 sans sous-requêtes parsées, fallback sur la question originale
    if difficulty == 3 and not sub_queries:
        sub_queries = [state["question"]]

    return {
        "sources": identified,
        "difficulty": difficulty,
        "initial_difficulty": difficulty,
        "current_query": state["question"],
        "sub_queries": sub_queries,
        "attempts": 0,
    }


def retrieve_node(state: AgentState) -> dict:
    """Lance le retrieval et construit le pool final selon le numéro de tentative.
    HyDE est désactivé pour les questions de difficulté 1 (économise un appel LLM).
    Pour le 1er retrieval d'une question décomposée (difficulté 3), on lance une
    requête par sous-question puis on fusionne et re-rank sur la question originale."""
    # HyDE activé uniquement au 1er retrieve des questions non-factuelles.
    # Au 2ème retrieve la query est déjà reformulée par rewrite_query → HyDE serait redondant.
    use_hyde = state["difficulty"] > 1 and state["attempts"] == 0

    if state["attempts"] == 0 and state["sub_queries"]:
        # Retrieval multi-sous-requêtes : HyDE désactivé car les sous-requêtes sont déjà
        # des phrases ciblées générées par le LLM — elles remplissent le même rôle que HyDE.
        all_docs: list[Document] = []
        seen_content: set[str] = set()
        pre_rerank: list[Document] = []
        pre_dedup: list[Document] = []
        semantic: list[Document] = []
        bm25: list[Document] = []
        dedup_pairs: list[tuple[Document, Document]] = []
        for sq in state["sub_queries"]:
            for doc in retrieve(sq, sources=state["sources"], verbose=False, use_hyde=False,
                                _pre_rerank_out=pre_rerank, _pre_dedup_out=pre_dedup,
                                _semantic_out=semantic, _bm25_out=bm25, _dedup_pairs_out=dedup_pairs):
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    all_docs.append(doc)
        # Re-rank global sur la question originale pour trier le pool fusionné
        final = _rerank(state["question"], all_docs) if len(all_docs) > K_FINAL else all_docs
        return {"docs": final, "attempts": 1, "pre_rerank_docs": pre_rerank,
                "pre_dedup_docs": pre_dedup, "semantic_docs": semantic, "bm25_docs": bm25,
                "dedup_removed_docs": [r for r, _ in dedup_pairs],
                "dedup_kept_docs": [k for _, k in dedup_pairs]}

    pre_rerank: list[Document] = []
    pre_dedup: list[Document] = []
    semantic: list[Document] = []
    bm25: list[Document] = []
    dedup_pairs: list[tuple[Document, Document]] = []
    new_docs = retrieve(state["current_query"], sources=state["sources"], verbose=False, use_hyde=use_hyde,
                        _pre_rerank_out=pre_rerank, _pre_dedup_out=pre_dedup,
                        _semantic_out=semantic, _bm25_out=bm25, _dedup_pairs_out=dedup_pairs)
    # Déduplique par contenu : évite d'envoyer deux fois le même chunk au LLM
    existing_contents = {d.page_content for d in state["docs"]}
    new_docs_deduped = [d for d in new_docs if d.page_content not in existing_contents]

    if state["attempts"] == 0:
        # 1er retrieval standard (difficulté 1 ou 2) : re-rank libre sur tous les chunks
        merged = state["docs"] + new_docs_deduped
        final = _rerank(state["question"], merged) if len(merged) > K_FINAL else merged
        return {"docs": final, "attempts": 1, "pre_rerank_docs": pre_rerank,
                "pre_dedup_docs": pre_dedup, "semantic_docs": semantic, "bm25_docs": bm25,
                "dedup_removed_docs": [r for r, _ in dedup_pairs],
                "dedup_kept_docs": [k for _, k in dedup_pairs]}

    # 2ème retrieval (post-rewrite) — 3 slots pour l'ancien pool + 2 slots réservés aux nouveaux chunks.
    # On garde une copie du pool précédent (state["docs"]) avant de l'écraser : c'est lui qui a servi
    # au grading initial et qui sert de baseline pour mesurer l'apport du rewrite (éval ⑤).
    old_top = state["docs"][:K_FINAL - 2]  # déjà triés par re-rank sur la question originale
    # Re-rank sur la question RÉÉCRITE : les nouveaux chunks ont été récupérés avec cette query,
    # c'est donc elle qui doit servir à sélectionner les 2 meilleurs.
    new_top = _rerank(state["current_query"], new_docs_deduped, n=2) if new_docs_deduped else []
    seen = {d.page_content for d in old_top}
    final = old_top + [d for d in new_top if d.page_content not in seen]

    return {"docs": final, "attempts": state["attempts"] + 1, "docs_before_rewrite": state["docs"], "post_rewrite_docs": new_docs}




def grade_documents(state: AgentState) -> dict:
    """Demande au LLM si les chunks accumulés suffisent, et ce qui manque précisément si non."""
    if not state["docs"]:
        return {"sufficient": False, "grade_verdict": "NON — aucun extrait récupéré"}
    context = "\n\n---\n\n".join(doc.page_content for doc in state["docs"])
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
    # Snapshot du grading qui a déclenché la reformulation, avant qu'il soit écrasé par le
    # grading final (post-rewrite) — sert de référence cohérente avec post_rerank_docs pour
    # l'évaluation du grading (④), pendant que sufficient/grade_verdict portent désormais le
    # verdict final post-rewrite, utilisé par l'évaluation du rewrite (⑤ rewrite_success).
    return {
        "current_query": new_query,
        "grading_before_rewrite": {"sufficient": state["sufficient"], "verdict": state["grade_verdict"]},
    }


def upgrade_difficulty(state: AgentState) -> dict:
    """Remonte la difficulté à 3 pour débloquer la reformulation.
    Appelé quand le grading (difficulté 2) juge les chunks insuffisants."""
    return {"difficulty": 3}


def generate_node(state: AgentState) -> dict:
    """Génère la réponse finale à partir du pool cumulatif de tous les chunks récupérés."""
    # Aucun chunk récupéré → inutile d'appeler le LLM
    _fallback = "Je ne trouve pas cette information dans les documents fournis."
    if not state["docs"]:
        return {"answer": _fallback}
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in state["docs"]
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=state["question"], context=context)
    answer = _invoke_with_retry(prompt).strip()
    # Le LLM peut renvoyer une chaîne vide (timeout silencieux, réponse tronquée…)
    return {"answer": answer if answer else _fallback}


def _route_after_retrieve(state: AgentState) -> str:
    """
    Difficulté 1 : generate directement (question factuelle, pas besoin de grading).
    Difficulté 2/3 : grade_documents, y compris après le dernier retrieve (post-rewrite) —
    ce grading final (qui ne peut plus déclencher de nouvelle boucle, cf. _route_after_grading)
    sert à enregistrer un verdict de suffisance sur le pool définitif, pour l'évaluateur (⑤).
    """
    if state["difficulty"] == 1:
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
        {"generate": "generate", "grade_documents": "grade_documents"},
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


def run_agent(question: str, verbose: bool = True) -> AgentState:
    """
    Exécute le graph agentique complet et retourne l'état final tel quel (tous les champs,
    y compris les champs de debug pre_rerank_docs/docs_before_rewrite) — utilisé par
    ask_question_agentic() ci-dessous et par run_agentic_all.py pour capturer tout ce dont
    l'évaluation par composant a besoin, sans jamais relancer le pipeline une 2e fois.
    """
    _maybe_restart_ollama(verbose=verbose)
    return agent.invoke({
        "question": question,
        "current_query": question,
        "sources": None,
        "difficulty": 2,
        "initial_difficulty": 2,  # écrasé par identify_sources
        "sub_queries": [],
        "docs": [],
        "sufficient": False,
        "grade_verdict": "",
        "attempts": 0,
        "answer": "",
        "pre_rerank_docs": [],
        "docs_before_rewrite": [],
        "post_rewrite_docs": [],
        "pre_dedup_docs": [],
        "semantic_docs": [],
        "bm25_docs": [],
        "dedup_removed_docs": [],
        "dedup_kept_docs": [],
        "grading_before_rewrite": {},
    })


def ask_question_agentic(question: str, verbose: bool = True) -> tuple[str, list[Document]]:
    """
    Pipeline RAG agentique complet : identification de la/les source(s) → retrieval →
    évaluation des chunks → (reformulation + nouveau retrieval si besoin, dans la limite
    de MAX_ATTEMPTS) → génération.
    Retourne un tuple (réponse_texte, chunks_utilisés), comme ask_question, pour rester
    interchangeable avec evaluate.py / app.py.
    """
    final_state = run_agent(question, verbose=verbose)

    if verbose:
        diff_labels = {1: "factuel", 2: "synthèse", 3: "complexe"}
        init_d = final_state["initial_difficulty"]
        final_d = final_state["difficulty"]
        upgrade_str = f" → remontée à 3 (pipeline complet)" if final_d > init_d else ""
        print(f"[Agent] Difficulté : {init_d} — {diff_labels.get(init_d, '?')}{upgrade_str}")
        if final_state["sub_queries"]:
            for i, sq in enumerate(final_state["sub_queries"], 1):
                print(f"[Agent] Sous-requête {i} : {sq}")
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
