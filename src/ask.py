from pathlib import Path  # Manipulation des chemins cross-platform

from langchain_chroma import Chroma  # Base de données vectorielle (stocke les embeddings)
from langchain_core.documents import Document  # Objet LangChain : texte + métadonnées
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Embedding + LLM via Ollama
from rank_bm25 import BM25Okapi  # Algorithme de recherche lexicale (par mots-clés)

# Chemins calculés depuis l'emplacement de ce fichier
BASE_DIR = Path(__file__).resolve().parent.parent         # Racine du projet
VECTOR_DB_DIR = BASE_DIR / "vector_db"                    # Base vectorielle Chroma sur disque
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"     # Template du prompt envoyé au LLM

# Paramètres de recherche
EMBED_MODEL = "bge-m3"    # Modèle d'embedding (doit être le même que celui utilisé dans ingest.py)
GEN_MODEL = "gemma2:2b"   # Modèle de génération de texte (LLM)
laK_RETRIEVE = 20           # Nombre de candidats récupérés par chaque méthode avant fusion
K_FINAL = 5               # Nombre de chunks finaux envoyés au LLM dans le contexte
RRF_K = 60                # Constante de la formule RRF (valeur standard = 60)

# Le LLM est instancié une fois au démarrage pour éviter de recharger le modèle à chaque question
llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096, temperature=0)  # temperature=0 = réponses déterministes

# Cache BM25 en mémoire : l'index n'est construit qu'une seule fois par session (coûteux)
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[tuple[str, dict]] | None = None


def _build_bm25_index(vector_db: Chroma) -> tuple[BM25Okapi, list[tuple[str, dict]]]:
    global _bm25_index, _bm25_chunks  # On modifie les variables de cache globales
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks  # Si déjà construit, on retourne directement le cache

    print("[BM25] Construction de l'index lexical (une fois par session)...")
    # Récupère tous les textes et métadonnées stockés dans Chroma
    result = vector_db._collection.get(include=["documents", "metadatas"])
    texts = result["documents"]   # Liste de tous les contenus textuels
    metas = result["metadatas"]   # Liste des métadonnées associées (source, page, etc.)

    _bm25_chunks = list(zip(texts, metas))  # Associe chaque texte à ses métadonnées
    # BM25 tokenise chaque texte en minuscules pour comparer avec la requête
    _bm25_index = BM25Okapi([t.lower().split() for t in texts])
    print(f"[BM25] Index prêt ({len(texts)} chunks).\n")
    return _bm25_index, _bm25_chunks


def _merge(
    semantic: list[tuple[Document, float]],  # Résultats de la recherche sémantique (doc + score cosinus)
    bm25_indices: list[int],                  # Indices BM25 triés par score décroissant
    bm25_chunks: list[tuple[str, dict]],      # Tous les chunks avec leurs métadonnées
    n: int = K_FINAL,                         # Nombre de chunks à retourner après fusion
) -> tuple[list[Document], list[tuple[str, float]]]:
    """Reciprocal Rank Fusion : score = Σ 1/(RRF_K + rang), indépendant des valeurs brutes."""
    scores: dict[str, float] = {}   # Accumule le score RRF pour chaque chunk (clé = contenu texte)
    doc_map: dict[str, Document] = {}  # Permet de retrouver l'objet Document depuis le texte

    # Contribution sémantique : chaque chunk reçoit 1/(60 + rang+1) selon sa position dans la liste
    for rank, (doc, _) in enumerate(semantic):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    # Contribution BM25 : même formule, mais basée sur le rang dans les résultats lexicaux
    for rank, idx in enumerate(bm25_indices):
        text, meta = bm25_chunks[idx]
        scores[text] = scores.get(text, 0) + 1.0 / (RRF_K + rank + 1)
        if text not in doc_map:  # Si ce chunk n'existait pas encore, on le crée
            doc_map[text] = Document(page_content=text, metadata=meta)

    # Trie tous les chunks par score RRF décroissant (les plus pertinents en premier)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Filtre de diversité : on n'accepte qu'un seul chunk par page source
    # Cela évite que le LLM reçoive 5 extraits du même paragraphe
    seen_pages: set[tuple] = set()
    top: list[tuple[str, float]] = []
    for key, score in ranked:
        meta = doc_map[key].metadata
        page_id = (meta.get("source"), meta.get("page"))  # Identifiant unique = (fichier, numéro de page)
        if page_id not in seen_pages:
            seen_pages.add(page_id)
            top.append((key, score))
        if len(top) == n:  # On s'arrête dès qu'on a assez de chunks diversifiés
            break

    return [doc_map[key] for key, _ in top], top  # Retourne les Documents et le classement avec scores


def _fmt(text: str, length: int = 130) -> str:
    # Utilitaire d'affichage : tronque et met sur une seule ligne pour les logs console
    return text[:length].replace("\n", " ")


def ask_question(question: str) -> str:
    # Charge les embeddings et ouvre la base vectorielle existante (créée par ingest.py)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
    print(f"\n[DB] {vector_db._collection.count()} chunks dans la base")

    # ── 1. Recherche sémantique (dense) ──────────────────────────────────
    # Convertit la question en vecteur et cherche les chunks les plus proches (distance cosinus)
    print(f"\n[Sémantique] Recherche des {K_RETRIEVE} plus proches voisins...")
    raw_semantic = vector_db.similarity_search_with_relevance_scores(question, k=K_RETRIEVE)

    print("[Sémantique] Top 5 :")
    for i, (doc, score) in enumerate(raw_semantic[:5]):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"  #{i+1}  score={score:.3f}  {src}  p.{page}")
        print(f"        ↳ {_fmt(doc.page_content)}")

    # ── 2. Recherche BM25 (lexicale / mots-clés) ─────────────────────────
    # BM25 est complémentaire : il trouve des chunks qui contiennent exactement les mêmes mots
    bm25, bm25_chunks = _build_bm25_index(vector_db)
    bm25_scores = bm25.get_scores(question.lower().split())  # Score BM25 pour chaque chunk
    # Trie les indices des chunks par score décroissant et garde les K_RETRIEVE meilleurs
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]

    print("\n[BM25] Top 5 résultats lexicaux :")
    for i, idx in enumerate(top_bm25[:5]):
        text, meta = bm25_chunks[idx]
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        print(f"  #{i+1}  bm25={bm25_scores[idx]:.2f}  {src}  p.{page}")
        print(f"        ↳ {_fmt(text)}")

    # ── 3. Fusion RRF ─────────────────────────────────────────────────────
    # Combine les deux listes en un seul classement hybride (sémantique + lexical)
    final_docs, rrf_ranking = _merge(raw_semantic, top_bm25, bm25_chunks)

    print(f"\n[Fusion] Top {K_FINAL} après fusion sémantique + BM25 :")
    for doc, (_, rrf_score) in zip(final_docs, rrf_ranking):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"  rrf={rrf_score:.4f}  {src}  p.{page}")
        print(f"        ↳ {_fmt(doc.page_content)}")

    if not final_docs:
        return "Je ne trouve pas cette information dans les documents fournis."

    # Assemble le contexte : on concatène les K_FINAL chunks avec leur source
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in final_docs
    )

    # Charge le template de prompt depuis le fichier et injecte la question + le contexte
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=question, context=context)
    return llm.invoke(prompt)  # Envoie le prompt au LLM et retourne sa réponse


if __name__ == "__main__":
    print("=== RAG — Mode terminal ===")
    print("Tapez votre question et appuyez sur Entrée. Ctrl+C pour quitter.\n")
    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue  # Ignore les entrées vides
            print("Recherche en cours...")
            answer = ask_question(question)
            print(f"\nRéponse :\n{answer}")
            print("\n" + "-" * 60 + "\n")
        except ConnectionError:
            print("\n[Erreur] Ollama n'est pas accessible. Vérifiez qu'il tourne (`ollama serve`).\n")
        except KeyboardInterrupt:
            print("\nAu revoir.")
            break
        except Exception as e:
            print(f"\n[Erreur inattendue] {type(e).__name__}: {e}\n")
