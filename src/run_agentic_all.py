import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # nécessaire sur Windows, pas disponible dans Jupyter
except AttributeError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import ask_question_agentic

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"
OUTPUT_PATH = DATASET_PATH.parent / "agentic_results.json"
PUSH_EVERY = 5  # sauvegarde sur GitHub toutes les N questions pour ne pas tout perdre si Colab coupe


class _Tee(io.TextIOBase):
    """Écrit en même temps sur la console et dans un buffer, pour capturer les logs [Agent]
    affichés par ask_question_agentic(verbose=True) tout en les gardant visibles à l'écran."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _git_push_results(n: int, total: int):
    """Push agentic_results.json sur GitHub pour ne pas perdre la progression si Colab coupe."""
    repo = OUTPUT_PATH.parent.parent  # racine du repo
    try:
        # Fetch + rebase AVANT de commiter pour éviter les conflits de rebase post-commit
        subprocess.run(["git", "-C", str(repo), "fetch", "origin", "main"], check=True)
        subprocess.run(["git", "-C", str(repo), "rebase", "origin/main"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", str(OUTPUT_PATH)], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", f"chore: sauvegarde résultats ({n}/{total})"], check=True)
        subprocess.run(["git", "-C", str(repo), "push"], check=True)
        print(f"  → pushé sur GitHub ({n}/{total})")
    except subprocess.CalledProcessError as e:
        print(f"  ! push échoué (non bloquant) : {e}")


def run_question(entry: dict) -> dict:
    """Lance le RAG agentique sur une question et regroupe tout ce qu'on veut garder :
    question, réponse attendue/générée, chunks récupérés et logs de l'agent."""
    question = entry["question"]
    print(f"\n{'='*60}")
    print(f"[{entry['id']}] {question}")
    print("=" * 60)

    buffer = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
        answer, docs = ask_question_agentic(question, verbose=True)
    duration = time.time() - start

    # Affiche les chunks sélectionnés avec leur score de re-ranking
    print(f"\n[Chunks sélectionnés] {len(docs)} chunk(s) :")
    for i, doc in enumerate(docs):
        score = doc.metadata.get("rerank_score")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        print(f"  #{i+1}  score={score_str}  {doc.metadata.get('source','?')}  p.{doc.metadata.get('page','?')}")

    return {
        "id": entry["id"],
        "question": question,
        "reponse_attendue": entry["reponse_attendue"],
        "reponse_llm": answer,
        "duree_secondes": round(duration, 1),
        "logs": buffer.getvalue().strip(),
        "chunks": [
            {
                "source": doc.metadata.get("source", "?"),
                "page": doc.metadata.get("page", "?"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "content": doc.page_content,
            }
            for doc in docs
        ],
    }


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    # Reprend là où on s'était arrêté (utile si le run de nuit est interrompu)
    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        already_done = {r["id"] for r in results}
    else:
        results = []
        already_done = set()

    to_run = [e for e in dataset if e["id"] not in already_done]

    print("=== RAG agentique — passage complet sur le dataset ===")
    print(f"Dataset      : {DATASET_PATH.name} ({len(dataset)} questions au total)")
    print(f"Déjà traités : {len(already_done)} — À traiter : {len(to_run)}")
    print(f"Résultats    : {OUTPUT_PATH}")

    try:
        for entry in to_run:
            result = run_question(entry)
            results.append(result)
            # Sauvegarde après chaque question : un crash ou un Ctrl+C en pleine nuit ne perd rien
            OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → sauvegardé ({len(results)}/{len(dataset)})")
            if len(results) % PUSH_EVERY == 0:
                _git_push_results(len(results), len(dataset))
    except KeyboardInterrupt:
        print("\n\nInterrompu — résultats partiels sauvegardés.")

    print(f"\nTerminé. {len(results)} question(s) traitée(s). Résultats dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
