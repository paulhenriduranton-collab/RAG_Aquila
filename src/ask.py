from pathlib import Path

from langchain_chroma import Chroma  # base de données vectorielle qui stocke les embeddings sur disque
from langchain_core.documents import Document  # objet LangChain : texte + métadonnées (source, page...)
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # connecteurs Ollama pour l'embedding et le LLM
from rank_bm25 import BM25Okapi  # algorithme de recherche lexicale par mots-clés
from sentence_transformers import CrossEncoder  # re-ranker : évalue chaque paire (question, chunk) ensemble

# Chemins calculés dynamiquement depuis l'emplacement de ce fichier
BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet (remonte 2 niveaux depuis src/)
VECTOR_DB_DIR = Path("C:/vector_db_aquila")        # hors OneDrive — SQLite corrompu par la synchro cloud
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"  # template du prompt envoyé au LLM

# Modèles utilisés — doivent être disponibles dans Ollama (ollama pull bge-m3 / gemma4:12b / gemma2:2b)
EMBED_MODEL = "bge-m3"    # modèle d'embedding multilingue — DOIT être le même que dans ingest.py
GEN_MODEL = "gemma4:12b"  # LLM qui génère la réponse finale à partir des chunks récupérés
HYDE_MODEL = "gemma2:2b"  # LLM léger pour HyDE — génère une réponse fictive, pas besoin du 12b
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # cross-encoder multilingue (HuggingFace)

# Paramètres du pipeline de retrieval
K_RETRIEVE = 20   # nombre de candidats récupérés par chaque méthode (sémantique ET BM25) avant fusion
K_RERANK = 10     # nombre de chunks passés au re-ranker après la fusion RRF
K_FINAL = 5       # nombre de chunks gardés après re-ranking — ce sont eux qui vont au LLM
RRF_K = 60        # constante de la formule RRF — valeur standard, ne pas changer sans raison

# Ces modèles sont instanciés une seule fois au démarrage du programme pour éviter de les recharger
llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096, temperature=0)        # temperature=0 = réponses déterministes (pas d'aléatoire)
hyde_llm = OllamaLLM(model=HYDE_MODEL, num_ctx=2048, temperature=0)  # modèle léger — HyDE n'a pas besoin d'un grand contexte
reranker = CrossEncoder(RERANK_MODEL)  # téléchargé automatiquement depuis HuggingFace au 1er lancement (~471 Mo)

# Variables globales pour le cache BM25 — l'index est coûteux à construire donc on le garde en mémoire
# Il est reconstruit uniquement si la session redémarre (None = pas encore construit)
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[tuple[str, dict]] | None = None


def _build_bm25_index(vector_db: Chroma, verbose: bool = True) -> tuple[BM25Okapi, list[tuple[str, dict]]]:
    """Construit l'index BM25 à partir de tous les chunks stockés dans Chroma (une fois par session)."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:  # si déjà construit lors d'une question précédente, on retourne le cache
        return _bm25_index, _bm25_chunks

    if verbose:
        print("[BM25] Construction de l'index lexical (une fois par session)...")

    # Récupère tous les textes et métadonnées stockés dans la base Chroma
    result = vector_db._collection.get(include=["documents", "metadatas"])
    texts = result["documents"]  # liste de tous les contenus textuels des chunks
    metas = result["metadatas"]  # liste des métadonnées associées (source, page, etc.)

    _bm25_chunks = list(zip(texts, metas))  # associe chaque texte à ses métadonnées pour pouvoir les retrouver
    _bm25_index = BM25Okapi([t.lower().split() for t in texts])  # tokenise chaque chunk en minuscules pour BM25

    if verbose:
        print(f"[BM25] Index prêt ({len(texts)} chunks).\n")
    return _bm25_index, _bm25_chunks


def _merge(
    semantic: list[tuple[Document, float]],  # résultats de la recherche sémantique : (doc, score cosinus)
    bm25_indices: list[int],                  # indices des meilleurs chunks BM25, triés par score décroissant
    bm25_chunks: list[tuple[str, dict]],      # tous les chunks avec leurs métadonnées
    n: int = K_RERANK,                        # nombre de chunks à retourner (K_RERANK = 10 par défaut)
    max_per_source: int = 5,                  # plafond de chunks par source — voir note ci-dessous
) -> tuple[list[Document], list[tuple[str, float]]]:
    """
    Reciprocal Rank Fusion (RRF) : combine les classements sémantique et BM25.
    Formule : score(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
    Avantage : indépendant des valeurs brutes des scores, ne regarde que les positions.

    max_per_source évite qu'un seul document monopolise les résultats lors d'une
    recherche sur toute la base (plusieurs PDF). Avec 2 sources et n=10, un plafond
    de 5 permet d'atteindre les 10 chunks voulus (5+5). Quand retrieve() restreint
    déjà la recherche à une ou plusieurs sources précises (cf. paramètre `sources`),
    tous les candidats partagent la même source : le plafond ne ferait alors que
    tronquer la liste à `max_per_source` éléments avant le re-ranking — c'est pour
    ça que retrieve() le désactive (= n) dans ce cas.
    """
    scores: dict[str, float] = {}   # accumule le score RRF pour chaque chunk (clé = contenu texte)
    doc_map: dict[str, Document] = {}  # permet de retrouver l'objet Document depuis le contenu texte

    # Contribution sémantique : chaque chunk reçoit 1/(60 + rang+1) selon sa position dans la liste
    for rank, (doc, _) in enumerate(semantic):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    # Contribution BM25 : même formule — les scores des deux méthodes s'additionnent
    # Un chunk bien classé dans les DEUX listes obtient un score cumulé élevé
    for rank, idx in enumerate(bm25_indices):
        text, meta = bm25_chunks[idx]
        scores[text] = scores.get(text, 0) + 1.0 / (RRF_K + rank + 1)
        if text not in doc_map:  # ce chunk n'était pas dans les résultats sémantiques
            doc_map[text] = Document(page_content=text, metadata=meta)

    # Trie tous les chunks par score RRF décroissant (les plus pertinents en premier)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Filtre de diversité : au plus `max_per_source` chunks par document source
    # Sans ce filtre, les 10 slots pourraient tous venir du même PDF
    source_count: dict[str, int] = {}
    top: list[tuple[str, float]] = []
    for key, score in ranked:
        source = doc_map[key].metadata.get("source", "?")
        if source_count.get(source, 0) < max_per_source:  # on accepte ce chunk si la source n'a pas encore atteint le plafond
            source_count[source] = source_count.get(source, 0) + 1
            top.append((key, score))
        if len(top) == n:  # on s'arrête quand on a assez de chunks diversifiés
            break

    return [doc_map[key] for key, _ in top], top


def _rerank(question: str, docs: list[Document], n: int = K_FINAL) -> list[Document]:
    """
    Re-classe les chunks avec un CrossEncoder plus précis que l'embedding.
    Contrairement à l'embedding (qui encode question et chunk séparément),
    le cross-encoder lit les deux textes ensemble et comprend mieux la pertinence.
    """
    # Forme des paires (question, chunk) — le cross-encoder a besoin des deux en même temps
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)  # retourne un score de pertinence pour chaque paire
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:n]]  # garde les n meilleurs


def _fmt(text: str, length: int = 130) -> str:
    """Utilitaire d'affichage : tronque le texte et le met sur une seule ligne pour les logs console."""
    return text[:length].replace("\n", " ")


# Prompt HyDE : demande au LLM une réponse fictive stylistiquement proche des brochures indexées,
# ce qui rapproche l'embedding de requête de l'espace des chunks-réponses (meilleure similarité cosinus).
HYDE_PROMPT = """Tu es un extrait de brochure universitaire. Réponds à cette question en 2-3 phrases courtes et factuelles, comme si tu étais le passage d'une brochure qui y répond directement :

{question}

Réponds directement, sans introduction ni guillemets."""


def _hyde(question: str) -> str:
    """Génère une réponse hypothétique (HyDE) pour améliorer la recherche sémantique."""
    return hyde_llm.invoke(HYDE_PROMPT.format(question=question)).strip()


def retrieve(question: str, sources: list[str] | None = None, verbose: bool = True) -> list[Document]:
    """
    Exécute le pipeline de retrieval complet sur une question :
    sémantique → BM25 → RRF → re-ranking.

    sources : si fourni, restreint la recherche aux chunks dont la métadonnée "source"
    (le nom du fichier, ex: "ENS.pdf") figure dans cette liste. Utilisé par le RAG
    agentique (agent.py) pour cibler un document précis et éviter les confusions
    inter-documents (ex: la durée de stage ENS vs Sorbonne).
    verbose=False désactive tous les logs (utile pour l'évaluation silencieuse).
    Retourne la liste des K_FINAL chunks les plus pertinents.
    """
    # Initialise la connexion à la base vectorielle existante (créée par ingest.py)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=embeddings)

    # Filtre Chroma natif sur la métadonnée "source" — None = pas de restriction (comportement inchangé)
    chroma_filter = {"source": {"$in": sources}} if sources else None

    if verbose:
        print(f"\n[DB] {vector_db._collection.count()} chunks dans la base")
        if sources:
            print(f"[Filtre] Recherche restreinte aux source(s) : {', '.join(sources)}")
        print(f"\n[HyDE] Génération de la réponse hypothétique...")

    # ── 1. Recherche sémantique (HyDE) ────────────────────────────────────────
    # On embed une réponse fictive plutôt que la question brute : une réponse est stylistiquement
    # plus proche des chunks indexés qu'une question, ce qui améliore la similarité cosinus.
    # BM25 (étape 2) continue d'utiliser la question originale pour les correspondances exactes.
    hyde_query = _hyde(question)
    if verbose:
        print(f"[HyDE] Réponse fictive : {_fmt(hyde_query)}")
        print(f"\n[Sémantique] Recherche des {K_RETRIEVE} plus proches voisins...")
    raw_semantic = vector_db.similarity_search_with_relevance_scores(hyde_query, k=K_RETRIEVE, filter=chroma_filter)

    if verbose:
        print("[Sémantique] Top 5 :")
        for i, (doc, score) in enumerate(raw_semantic[:5]):
            print(f"  #{i+1}  score={score:.3f}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")
            print(f"        ↳ {_fmt(doc.page_content)}")

    # ── 2. Recherche BM25 (lexicale) ─────────────────────────────────────────
    # Complémentaire à la sémantique : trouve les chunks contenant exactement les mêmes mots
    bm25, bm25_chunks = _build_bm25_index(vector_db, verbose=verbose)
    bm25_scores = bm25.get_scores(question.lower().split())  # score BM25 pour chaque chunk de la base
    # Restreint les candidats aux sources demandées avant de trier (même filtre que côté sémantique)
    candidate_indices = range(len(bm25_chunks))
    if sources:
        candidate_indices = [i for i in candidate_indices if bm25_chunks[i][1].get("source") in sources]
    # Trie les indices par score décroissant et garde les K_RETRIEVE meilleurs
    top_bm25 = sorted(candidate_indices, key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]

    if verbose:
        print("\n[BM25] Top 5 résultats lexicaux :")
        for i, idx in enumerate(top_bm25[:5]):
            text, meta = bm25_chunks[idx]
            print(f"  #{i+1}  bm25={bm25_scores[idx]:.2f}  {meta.get('source','?')}  p.{meta.get('page','?')}")
            print(f"        ↳ {_fmt(text)}")

    # ── 3. Fusion RRF ─────────────────────────────────────────────────────────
    # Combine les deux classements (sémantique + BM25) en un seul classement hybride.
    # Si `sources` restreint déjà la recherche, tous les candidats partagent la même
    # source : on désactive le plafond de diversité (max_per_source = K_RERANK, donc
    # sans effet) pour ne pas tronquer la liste à 3 chunks avant le re-ranking.
    rrf_docs, rrf_ranking = _merge(
        raw_semantic, top_bm25, bm25_chunks,
        max_per_source=K_RERANK if sources else 3,
    )

    if verbose:
        print(f"\n[RRF] Top {K_RERANK} après fusion sémantique + BM25 :")
        for doc, (_, rrf_score) in zip(rrf_docs, rrf_ranking):
            print(f"  rrf={rrf_score:.4f}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")
            print(f"        ↳ {_fmt(doc.page_content)}")

    if not rrf_docs:
        return []

    # ── 4. Re-ranking ─────────────────────────────────────────────────────────
    # Le cross-encoder reclasse les K_RERANK candidats RRF avec une lecture conjointe (question + chunk)
    final_docs = _rerank(question, rrf_docs)

    if verbose:
        print(f"\n[Top {K_FINAL} final] :")
        for i, doc in enumerate(final_docs):
            print(f"  #{i+1}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")
            print(f"        ↳ {_fmt(doc.page_content)}")

    return final_docs


def ask_question(question: str, verbose: bool = True) -> tuple[str, list[Document]]:
    """
    Pipeline RAG complet : retrieval + génération.
    Retourne un tuple (réponse_texte, chunks_utilisés).

    verbose=False désactive tous les logs (utilisé par evaluate.py et app.py).
    """
    final_docs = retrieve(question, verbose=verbose)

    if not final_docs:
        return "Je ne trouve pas cette information dans les documents fournis.", []

    # Assemble le contexte : concatène les chunks avec leur source pour que le LLM sache d'où vient chaque info
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in final_docs
    )

    # Charge le template de prompt et injecte la question + le contexte
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=question, context=context)
    return llm.invoke(prompt), final_docs  # retourne la réponse ET les chunks pour l'évaluation


if __name__ == "__main__":
    # Mode terminal : boucle interactive pour poser des questions et voir les logs de retrieval
    print("=== RAG — Mode terminal ===")
    print("Tapez votre question et appuyez sur Entrée. Ctrl+C pour quitter.\n")
    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue
            print("Recherche en cours...")
            answer, _ = ask_question(question)  # on ignore les chunks retournés en mode terminal
            print(f"\nRéponse :\n{answer}")
            print("\n" + "-" * 60 + "\n")
        except ConnectionError:
            print("\n[Erreur] Ollama n'est pas accessible. Vérifiez qu'il tourne (`ollama serve`).\n")
        except KeyboardInterrupt:
            print("\nAu revoir.")
            break
        except Exception as e:
            print(f"\n[Erreur inattendue] {type(e).__name__}: {e}\n")
