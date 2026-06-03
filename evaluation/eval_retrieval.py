"""
Évaluation du retrieval sur le dataset synthétique.

Métriques calculées :
  - Recall@K  : parmi les K chunks retournés, le bon chunk est-il présent ?
  - MRR       : à quelle position moyenne apparaît le bon chunk ?

Usage :
    python evaluation/eval_retrieval.py

    # Tester un jeu de paramètres différent sans modifier ask.py :
    python evaluation/eval_retrieval.py --k-retrieve 30 --bm25-weight 0.3 --threshold 0.45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
VECTOR_DB   = BASE_DIR / "vector_db"
DATASET     = Path(__file__).parent / "dataset.json"
EMBED_MODEL = "nomic-embed-text"

# ── Paramètres par défaut (identiques à ask.py) ───────────────────────────────
DEFAULT_K_RETRIEVE  = 20
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_THRESHOLD   = 0.50
EVAL_DEPTH          = 10   # on mesure jusqu'au rang 10


# ── Utilitaires ───────────────────────────────────────────────────────────────

def build_text_to_id(db: Chroma) -> dict[str, str]:
    """Construit un mapping texte → chunk_id ChromaDB."""
    raw = db._collection.get(include=["documents"])
    return {text: cid for cid, text in zip(raw["ids"], raw["documents"])}


def build_bm25(db: Chroma) -> tuple[BM25Okapi, list[tuple[str, str, dict]]]:
    """Construit l'index BM25 sur tous les chunks. Retourne (index, [(id, text, meta)])."""
    raw    = db._collection.get(include=["documents", "metadatas"])
    ids    = raw["ids"]
    texts  = raw["documents"]
    metas  = raw["metadatas"]
    index  = BM25Okapi([t.lower().split() for t in texts])
    chunks = list(zip(ids, texts, metas))
    return index, chunks


def retrieve(
    question: str,
    db: Chroma,
    text_to_id: dict[str, str],
    bm25_index: BM25Okapi,
    bm25_chunks: list[tuple[str, str, dict]],
    k_retrieve: int,
    bm25_weight: float,
    threshold: float,
    depth: int,
) -> list[str]:
    """
    Réplique exactement le pipeline de ask.py (fusion sémantique + BM25).
    Retourne la liste ordonnée des chunk_ids (du plus pertinent au moins pertinent).
    Note : le filtre de diversité (1 chunk par page) n'est PAS appliqué ici
    afin d'obtenir une mesure pure de la qualité du ranking.
    """
    # ── 1. Recherche sémantique ───────────────────────────────────────────────
    semantic_raw = db.similarity_search_with_relevance_scores(question, k=k_retrieve)

    scores: dict[str, float] = {}
    for doc, score in semantic_raw:
        if score < threshold:
            continue
        key = doc.page_content
        scores[key] = scores.get(key, 0) + score

    # ── 2. Recherche BM25 ─────────────────────────────────────────────────────
    bm25_scores = bm25_index.get_scores(question.lower().split())
    top_bm25    = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_retrieve]
    max_bm25    = max((bm25_scores[i] for i in top_bm25), default=1.0)

    for idx in top_bm25:
        _, text, _ = bm25_chunks[idx]
        scores[text] = scores.get(text, 0) + (bm25_scores[idx] / max_bm25) * bm25_weight

    # ── 3. Classement final ───────────────────────────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result_ids: list[str] = []
    for text, _ in ranked[:depth]:
        cid = text_to_id.get(text)
        if cid:
            result_ids.append(cid)

    return result_ids


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    # ── Chargement du dataset ─────────────────────────────────────────────────
    if not DATASET.exists():
        print(f"Dataset introuvable : {DATASET}")
        print("Générez-le d'abord : python evaluation/generate_dataset.py")
        return

    with open(DATASET, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    if not dataset:
        print("Le dataset est vide.")
        return

    print(f"Dataset : {len(dataset)} questions chargées.")
    print(f"Paramètres : K_RETRIEVE={args.k_retrieve}  BM25_WEIGHT={args.bm25_weight}  THRESHOLD={args.threshold}\n")

    # ── Connexion ChromaDB + construction des index ───────────────────────────
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db         = Chroma(persist_directory=str(VECTOR_DB), embedding_function=embeddings)
    print("Construction des index (BM25 + mapping texte→id)…")
    text_to_id              = build_text_to_id(db)
    bm25_index, bm25_chunks = build_bm25(db)
    print(f"{len(text_to_id)} chunks indexés.\n")

    # ── Boucle d'évaluation ───────────────────────────────────────────────────
    records: list[dict] = []

    for i, entry in enumerate(dataset):
        q      = entry["question"]
        target = entry["chunk_id"]
        src    = entry["source"]
        print(f"  [{i+1:3d}/{len(dataset)}]  {q[:65]}… ", end="", flush=True)

        retrieved = retrieve(
            question    = q,
            db          = db,
            text_to_id  = text_to_id,
            bm25_index  = bm25_index,
            bm25_chunks = bm25_chunks,
            k_retrieve  = args.k_retrieve,
            bm25_weight = args.bm25_weight,
            threshold   = args.threshold,
            depth       = EVAL_DEPTH,
        )

        rank: int | None = None
        for j, cid in enumerate(retrieved):
            if cid == target:
                rank = j + 1
                break

        records.append({"question": q, "source": src, "rank": rank})

        if rank is None:
            print("✗  non trouvé")
        else:
            print(f"✓  rang {rank}")

    # ── Calcul des métriques ──────────────────────────────────────────────────
    n = len(records)
    ks = [1, 3, 5, 10]

    recall: dict[int, float] = {
        k: sum(1 for r in records if r["rank"] is not None and r["rank"] <= k) / n
        for k in ks
    }
    mrr = sum(1 / r["rank"] for r in records if r["rank"] is not None) / n

    print(f"\n{'='*55}")
    print(f"  RÉSULTATS GLOBAUX  ({n} questions)")
    print(f"{'='*55}")
    for k in ks:
        bar  = "█" * round(recall[k] * 20)
        pad  = "░" * (20 - len(bar))
        flag = "  ← objectif minimum" if k == 5 else ""
        print(f"  Recall@{k:<3}  {bar}{pad}  {recall[k]:.1%}{flag}")
    print(f"\n  MRR          {mrr:.3f}   (1.0 = toujours en 1ère position)")

    # ── Détail par document ───────────────────────────────────────────────────
    sources = sorted({r["source"] for r in records})
    if len(sources) > 1:
        print(f"\n  Par document :")
        for src in sources:
            sub     = [r for r in records if r["source"] == src]
            r5      = sum(1 for r in sub if r["rank"] is not None and r["rank"] <= 5) / len(sub)
            sub_mrr = sum(1 / r["rank"] for r in sub if r["rank"] is not None) / len(sub)
            print(f"    {src:<42}  Recall@5={r5:.1%}  MRR={sub_mrr:.3f}  ({len(sub)} questions)")

    # ── Questions non trouvées ────────────────────────────────────────────────
    missed = [r for r in records if r["rank"] is None]
    if missed:
        print(f"\n  Questions non trouvées ({len(missed)}) :")
        for r in missed:
            print(f"    – {r['question'][:85]}")

    # ── Interprétation ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Interprétation rapide :")
    if recall[5] < 0.40:
        print("  ► Recall@5 < 40% → problème sérieux de retrieval.")
        print("    Pistes : augmenter chunk_overlap, changer le modèle d'embedding,")
        print("             remonter K_RETRIEVE, abaisser le seuil sémantique.")
    elif recall[5] < 0.65:
        print("  ► Recall@5 correct mais améliorable.")
        print("    Pistes : tuner BM25_WEIGHT, ajouter un reranker cross-encoder.")
    else:
        print("  ► Recall@5 > 65% — le retrieval est solide.")
        print("    Si les réponses restent mauvaises, le goulot est dans la génération.")
        print("    Pistes : changer de LLM, enrichir le prompt, augmenter K_FINAL.")

    if mrr < 0.30:
        print("  ► MRR faible : le bon chunk est souvent trouvé mais mal classé.")
        print("    Piste : implémenter le vrai RRF (1/(60+rang)) ou un reranker.")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation du retrieval RAG")
    p.add_argument("--k-retrieve",  type=int,   default=DEFAULT_K_RETRIEVE,  help="Candidats récupérés par méthode")
    p.add_argument("--bm25-weight", type=float, default=DEFAULT_BM25_WEIGHT, help="Poids du score BM25 dans la fusion")
    p.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD,   help="Score sémantique minimum")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
