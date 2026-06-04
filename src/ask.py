from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
PROMPT_PATH = BASE_DIR / "prompts" / "rag_prompt.txt"

EMBED_MODEL = "bge-m3"
GEN_MODEL = "gemma2:2b"
K_RETRIEVE = 20   # candidats récupérés par chaque méthode avant fusion
K_FINAL = 5       # chunks envoyés au LLM
RRF_K = 60        # constante RRF (standard = 60)

llm = OllamaLLM(model=GEN_MODEL, num_ctx=4096, temperature=0)

# Index BM25 construit une seule fois par session (coûteux à reconstruire)
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[tuple[str, dict]] | None = None


def _build_bm25_index(vector_db: Chroma) -> tuple[BM25Okapi, list[tuple[str, dict]]]:
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    print("[BM25] Construction de l'index lexical (une fois par session)...")
    result = vector_db._collection.get(include=["documents", "metadatas"])
    texts = result["documents"]
    metas = result["metadatas"]
    _bm25_chunks = list(zip(texts, metas))
    _bm25_index = BM25Okapi([t.lower().split() for t in texts])
    print(f"[BM25] Index prêt ({len(texts)} chunks).\n")
    return _bm25_index, _bm25_chunks


def _merge(
    semantic: list[tuple[Document, float]],
    bm25_indices: list[int],
    bm25_chunks: list[tuple[str, dict]],
    n: int = K_FINAL,
) -> tuple[list[Document], list[tuple[str, float]]]:
    """Reciprocal Rank Fusion : score = Σ 1/(RRF_K + rang), indépendant des valeurs brutes."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(semantic):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        doc_map[key] = doc

    for rank, idx in enumerate(bm25_indices):
        text, meta = bm25_chunks[idx]
        scores[text] = scores.get(text, 0) + 1.0 / (RRF_K + rank + 1)
        if text not in doc_map:
            doc_map[text] = Document(page_content=text, metadata=meta)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Diversité : max 1 chunk par page pour éviter de saturer le contexte avec la même source
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


def _fmt(text: str, length: int = 130) -> str:
    return text[:length].replace("\n", " ")


def ask_question(question: str) -> str:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )
    print(f"\n[DB] {vector_db._collection.count()} chunks dans la base")

    # ── 1. Recherche sémantique ───────────────────────────────────────────
    print(f"\n[Sémantique] Recherche des {K_RETRIEVE} plus proches voisins...")
    raw_semantic = vector_db.similarity_search_with_relevance_scores(question, k=K_RETRIEVE)

    print("[Sémantique] Top 5 :")
    for i, (doc, score) in enumerate(raw_semantic[:5]):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"  #{i+1}  score={score:.3f}  {src}  p.{page}")
        print(f"        ↳ {_fmt(doc.page_content)}")

    # ── 2. Recherche BM25 (mots-clés) ────────────────────────────────────
    bm25, bm25_chunks = _build_bm25_index(vector_db)
    bm25_scores = bm25.get_scores(question.lower().split())
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K_RETRIEVE]

    print("\n[BM25] Top 5 résultats lexicaux :")
    for i, idx in enumerate(top_bm25[:5]):
        text, meta = bm25_chunks[idx]
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        print(f"  #{i+1}  bm25={bm25_scores[idx]:.2f}  {src}  p.{page}")
        print(f"        ↳ {_fmt(text)}")

    # ── 3. Fusion RRF ─────────────────────────────────────────────────────
    final_docs, rrf_ranking = _merge(raw_semantic, top_bm25, bm25_chunks)

    print(f"\n[Fusion] Top {K_FINAL} après fusion sémantique + BM25 :")
    for doc, (_, rrf_score) in zip(final_docs, rrf_ranking):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"  rrf={rrf_score:.4f}  {src}  p.{page}")
        print(f"        ↳ {_fmt(doc.page_content)}")

    if not final_docs:
        return "Je ne trouve pas cette information dans les documents fournis."

    context = "\n\n---\n\n".join(
        f"Source : {doc.metadata.get('source', '?')}\n{doc.page_content}"
        for doc in final_docs
    )

    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(question=question, context=context)
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
        except ConnectionError:
            print("\n[Erreur] Ollama n'est pas accessible. Vérifiez qu'il tourne (`ollama serve`).\n")
        except KeyboardInterrupt:
            print("\nAu revoir.")
            break
        except Exception as e:
            print(f"\n[Erreur inattendue] {type(e).__name__}: {e}\n")
