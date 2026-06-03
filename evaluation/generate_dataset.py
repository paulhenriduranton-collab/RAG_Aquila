"""
Génération du dataset d'évaluation synthétique.

Pour chaque chunk de la base vectorielle, un LLM génère une question
dont la réponse se trouve explicitement dans ce chunk.

Usage :
    python evaluation/generate_dataset.py

Sortie :
    evaluation/dataset.json  →  à relire et nettoyer manuellement avant d'évaluer
"""

from pathlib import Path
import json
import random

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR    = Path(__file__).resolve().parent.parent
VECTOR_DB   = BASE_DIR / "vector_db"
OUTPUT      = Path(__file__).parent / "dataset.json"

EMBED_MODEL       = "nomic-embed-text"
GEN_MODEL         = "gemma2:2b"
CHUNKS_PER_SOURCE = 20   # nombre de chunks échantillonnés par PDF
MIN_CHUNK_LEN     = 300  # ignorer les chunks trop courts (entêtes, pages vides…)
RANDOM_SEED       = 42

PROMPT = """\
Tu es un professeur de mathématiques qui prépare des questions d'examen.
À partir du texte ci-dessous, génère UNE SEULE question précise en français.

Contraintes strictes :
- La réponse doit se trouver EXPLICITEMENT dans le texte fourni
- La question doit être naturelle, comme celle d'un étudiant préparant un examen
- Elle doit porter sur une définition, un théorème, une propriété ou une étape de démonstration
- Écris UNIQUEMENT la question, sans introduction, sans numérotation, sans explication

TEXTE :
{chunk}

QUESTION :"""


def main() -> None:
    random.seed(RANDOM_SEED)

    print("Connexion à ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db  = Chroma(persist_directory=str(VECTOR_DB), embedding_function=embeddings)
    llm = OllamaLLM(model=GEN_MODEL, temperature=0.3, num_ctx=2048)

    # ── Récupérer tous les chunks avec leurs IDs ─────────────────────────────
    raw    = db._collection.get(include=["documents", "metadatas"])
    chunks = list(zip(raw["ids"], raw["documents"], raw["metadatas"]))
    print(f"{len(chunks)} chunks dans la base.\n")

    # ── Grouper par source et filtrer les chunks trop courts ──────────────────
    by_source: dict[str, list] = {}
    for chunk_id, text, meta in chunks:
        if len(text) < MIN_CHUNK_LEN:
            continue
        src = meta.get("source", "inconnu")
        by_source.setdefault(src, []).append((chunk_id, text, meta))

    # ── Échantillonnage équilibré ─────────────────────────────────────────────
    sampled: list[tuple] = []
    for src, src_chunks in sorted(by_source.items()):
        n = min(CHUNKS_PER_SOURCE, len(src_chunks))
        selected = random.sample(src_chunks, n)
        sampled.extend(selected)
        print(f"  {src:<45}  {n} chunks sélectionnés")

    total = len(sampled)
    print(f"\nGénération de {total} questions (cela peut prendre quelques minutes)...\n")

    dataset = []
    skipped = 0

    for i, (chunk_id, chunk_text, meta) in enumerate(sampled):
        src  = meta.get("source", "?")
        page = meta.get("page", "?")
        print(f"  [{i+1:3d}/{total}]  {src}  p.{page} … ", end="", flush=True)

        try:
            raw_answer = llm.invoke(PROMPT.format(chunk=chunk_text[:1200])).strip()

            # Nettoyage : enlever les préfixes parasites que le LLM peut ajouter
            for prefix in ("Question :", "Question:", "Q :", "Q:", "- "):
                if raw_answer.lower().startswith(prefix.lower()):
                    raw_answer = raw_answer[len(prefix):].strip()

            # Validation minimale : doit ressembler à une vraie question
            if "?" not in raw_answer or not (15 <= len(raw_answer) <= 350):
                print("⚠  ignoré (qualité insuffisante)")
                skipped += 1
                continue

            dataset.append({
                "id":         len(dataset) + 1,
                "question":   raw_answer,
                "chunk_id":   chunk_id,
                "chunk_text": chunk_text,
                "source":     src,
                "page":       int(page) if str(page).isdigit() else page,
            })
            print(f"✓  {raw_answer[:70]}…")

        except Exception as exc:
            print(f"✗  erreur : {exc}")
            skipped += 1

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Dataset sauvegardé : {OUTPUT}")
    print(f"  {len(dataset)} paires générées  |  {skipped} ignorées")
    print(f"{'='*60}")
    print("""
Prochaines étapes :
  1. Ouvrez evaluation/dataset.json
  2. Parcourez les questions et supprimez celles de mauvaise qualité
     (trop vagues, hors-sujet, incompréhensibles sans le chunk)
  3. Conservez idéalement 25 à 40 questions de bonne qualité
  4. Lancez : python evaluation/eval_retrieval.py
""")


if __name__ == "__main__":
    main()
