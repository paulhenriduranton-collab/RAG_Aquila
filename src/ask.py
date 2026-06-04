from pathlib import Path

from langchain_chroma import Chroma  # base de données vectorielle
from langchain_core.documents import Document  # objet texte + métadonnées
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # embedding et LLM via Ollama
from rank_bm25 import BM25Okapi  # recherche par mots-clés
from sentence_transformers import CrossEncoder  # re-ranker : note chaque paire (question, chunk)

BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
VECTOR_DB_DIR = BASE_DIR / "vector_db"
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"

EMBED_MODEL = "bge-m3"   # même modèle que dans ingest.py
GEN_MODEL = "gemma2:2b"  # LLM qui génère la réponse finale
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # cross-encoder multilingue
K_RETRIEVE = 20   # candidats récupérés par chaque méthode avant fusion
K_RERANK = 10     # chunks passés au re-ranker après fusion RRF
K_FINAL = 5       # chunks gardés après re-ranking, envoyés au LLM
RRF_K = 60        # constante de la formule RRF (valeur standard)

# Modèles instanciés une fois au démarrage
llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096, temperature=0)  # temperature=0 = réponses déterministes
reranker = CrossEncoder(RERANK_MODEL)  # chargé localement depuis HuggingFace (téléchargé au 1er lancement)

# Cache global : l'index BM25 est coûteux à construire, on le garde en mémoire
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[tuple[str, dict]] | None = None


def _build_bm25_index(vector_db: Chroma, verbose: bool = True) -> tuple[BM25Okapi, list[tuple[str, dict]]]:
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:  # déjà construit lors d'une question précédente
        return _bm25_index, _bm25_chunks

    if verbose:
        print("[BM25] Construction de l'index lexical (une fois par session)...")
    result = vector_db._collection.get(include=["documents", "metadatas"])
    texts = result["documents"]
    metas = result["metadatas"]
    _bm25_chunks = list(zip(texts, metas))
    _bm25_index = BM25Okapi([t.lower().split() for t in texts])  # tokenisation simple en minuscules
    if verbose:
        print(f"[BM25] Index prêt ({len(texts)} chunks).\n")
    return _bm25_index, _bm25_chunks


def _merge(
    semantic: list[tuple[Document, float]],  # résultats sémantiques (doc + score cosinus)
    bm25_indices: list[int],                  # indices BM25 triés par score
    bm25_chunks: list[tuple[str, dict]],
    n: int = K_RERANK,  # on récupère K_RERANK candidats pour le re-ranker
) -> tuple[list[Document], list[tuple[str, float]]]:
    """Reciprocal Rank Fusion : score = Σ 1/(RRF_K + rang), indépendant des valeurs brutes."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Chaque chunk reçoit un score basé sur sa position dans les résultats sémantiques
    for rank, (doc, _) in enumerate(semantic):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    # Même formule pour les résultats BM25 — les scores des deux méthodes s'additionnent
    for rank, idx in enumerate(bm25_indices):
        text, meta = bm25_chunks[idx]
        scores[text] = scores.get(text, 0) + 1.0 / (RRF_K + rank + 1)
        if text not in doc_map:
            doc_map[text] = Document(page_content=text, metadata=meta)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Filtre de diversité : max 1 chunk par page source pour varier les extraits
    seen_pages: set[tuple] = set()
    top: list[tuple[str, float]] = []
    for key, score in ranked:
        meta = doc_map[key].metadata
        page_id = (meta.get("source"), meta.get("page"))
        if page_id not in seen_pages:
            seen_pages.add(page_id)
            top.append((key, score))
        if len(top) == n:
            break

    return [doc_map[key] for key, _ in top], top


def _rerank(question: str, docs: list[Document], n: int = K_FINAL) -> list[Document]:
    # Le cross-encoder évalue chaque paire (question, chunk) ensemble — plus précis que l'embedding
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)  # retourne un score de pertinence pour chaque paire
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:n]]


def _fmt(text: str, length: int = 130) -> str:
    # Tronque et met sur une ligne pour l'affichage console
    return text[:length].replace("\n", " ")


def retrieve(question: str, verbose: bool = True) -> list[Document]:
    """Retourne les chunks finaux après sémantique + BM25 + RRF + re-ranking."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=embeddings)

    if verbose:
        print(f"\n[DB] {vector_db._collection.count()} chunks dans la base")
        print(f"\n[Sémantique] Recherche des {K_RETRIEVE} plus proches voisins...")

    raw_semantic = vector_db.similarity_search_with_relevance_scores(question, k=K_RETRIEVE)

    if verbose:
        print("[Sémantique] Top 5 :")
        for i, (doc, score) in enumerate(raw_semantic[:5]):
            print(f"  #{i+1}  score={score:.3f}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")
            print(f"        ↳ {_fmt(doc.page_content)}")

    # ── 2. Recherche BM25 ────────────────────────────────────────────────────────────────────────
    bm25, bm25_chunks = _build_bm25_index(vector_db, verbose=verbose)
    bm25_scores = bm25.get_scores(question.lower().split())
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]

    if verbose:
        print("\n[BM25] Top 5 résultats lexicaux :")
        for i, idx in enumerate(top_bm25[:5]):
            text, meta = bm25_chunks[idx]
            print(f"  #{i+1}  bm25={bm25_scores[idx]:.2f}  {meta.get('source','?')}  p.{meta.get('page','?')}")
            print(f"        ↳ {_fmt(text)}")

    # ── 3. Fusion RRF ────────────────────────────────────────────────────────────────────────────
    rrf_docs, rrf_ranking = _merge(raw_semantic, top_bm25, bm25_chunks)

    if verbose:
        print(f"\n[RRF] Top {K_RERANK} après fusion sémantique + BM25 :")
        for doc, (_, rrf_score) in zip(rrf_docs, rrf_ranking):
            print(f"  rrf={rrf_score:.4f}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")
            print(f"        ↳ {_fmt(doc.page_content)}")

    if not rrf_docs:
        return []

    # ── 4. Re-ranking ────────────────────────────────────────────────────────────────────────────
    final_docs = _rerank(question, rrf_docs)

    if verbose:
        print(f"\n[Top {K_FINAL} final] :")
        for i, doc in enumerate(final_docs):
            print(f"  #{i+1}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")
            print(f"        ↳ {_fmt(doc.page_content)}")

    return final_docs


def ask_question(question: str, verbose: bool = True) -> tuple[str, list[Document]]:
    """Retourne (réponse, chunks_utilisés)."""
    final_docs = retrieve(question, verbose=verbose)

    if not final_docs:
        return "Je ne trouve pas cette information dans les documents fournis.", []

    # Assemble le contexte à envoyer au LLM (chunks + leur source)
    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in final_docs
    )

    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=question, context=context)
    return llm.invoke(prompt), final_docs


if __name__ == "__main__":
    print("=== RAG — Mode terminal ===")
    print("Tapez votre question et appuyez sur Entrée. Ctrl+C pour quitter.\n")
    while True:
        try:
            question = input("Question : ").strip()
            if not question:
                continue
            print("Recherche en cours...")
            answer, _ = ask_question(question)
            print(f"\nRéponse :\n{answer}")
            print("\n" + "-" * 60 + "\n")
        except ConnectionError:
            print("\n[Erreur] Ollama n'est pas accessible. Vérifiez qu'il tourne (`ollama serve`).\n")
        except KeyboardInterrupt:
            print("\nAu revoir.")
            break
        except Exception as e:
            print(f"\n[Erreur inattendue] {type(e).__name__}: {e}\n")
