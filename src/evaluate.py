import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ask_question, llm, GEN_MODEL, EMBED_MODEL

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.json"

# ── Prompts d'évaluation envoyés au LLM ──────────────────────────────────────

PROMPT_FAITHFULNESS = """Tu es un évaluateur. Lis le contexte et la réponse ci-dessous.
Réponds uniquement par un nombre entre 0.0 et 1.0.
1.0 = toutes les affirmations de la réponse sont présentes dans le contexte (pas d'invention).
0.0 = la réponse contient des informations absentes du contexte (hallucination).

Contexte :
{context}

Réponse :
{answer}

Score (0.0 à 1.0) :"""

PROMPT_RELEVANCY = """Tu es un évaluateur. Lis la question et la réponse ci-dessous.
Réponds uniquement par un nombre entre 0.0 et 1.0.
1.0 = la réponse répond directement et complètement à la question.
0.0 = la réponse ne répond pas à la question.

Question :
{question}

Réponse :
{answer}

Score (0.0 à 1.0) :"""

PROMPT_CONTEXT_QUALITY = """Tu es un évaluateur. Lis la question et les extraits récupérés ci-dessous.
Réponds uniquement par un nombre entre 0.0 et 1.0.
1.0 = tous les extraits sont utiles pour répondre à la question.
0.0 = les extraits sont hors sujet.

Question :
{question}

Extraits :
{context}

Score (0.0 à 1.0) :"""

PROMPT_CONTEXT_RECALL = """Tu es un évaluateur. Compare la réponse de référence et les extraits récupérés.
Réponds uniquement par un nombre entre 0.0 et 1.0.
1.0 = tous les éléments de la réponse de référence sont couverts par les extraits.
0.0 = les extraits ne contiennent aucune information de la réponse de référence.

Réponse de référence :
{ground_truth}

Extraits récupérés :
{context}

Score (0.0 à 1.0) :"""

PROMPT_ANSWER_CORRECTNESS = """Tu es un évaluateur. Compare la réponse générée et la réponse de référence.
Réponds uniquement par un nombre entre 0.0 et 1.0.
1.0 = la réponse générée est factuellement identique à la référence.
0.0 = la réponse générée est fausse ou contradictoire avec la référence.

Réponse de référence :
{ground_truth}

Réponse générée :
{answer}

Score (0.0 à 1.0) :"""


def _score(prompt: str) -> float:
    """Envoie le prompt au LLM et extrait le score numérique."""
    raw = llm.invoke(prompt).strip()
    for token in raw.replace(",", ".").split():
        try:
            return max(0.0, min(1.0, float(token)))
        except ValueError:
            continue
    return 0.0


def evaluate_question(entry: dict) -> dict | None:
    """Lance le pipeline RAG sur une question et calcule les 5 métriques."""
    question     = entry["question"]
    ground_truth = entry["reponse_attendue"]

    print(f"\n{'='*60}")
    print(f"[{entry['id']}] Niveau {entry['niveau']} — {entry['type']}")
    print(f"Question : {question}")
    print("="*60)

    answer, docs = ask_question(question, verbose=False)
    if not docs:
        print("  ⚠ Aucun document trouvé, question ignorée.")
        return None

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    print(f"Réponse  : {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print("Évaluation en cours...")

    # Les 3 métriques sans ground truth
    faithfulness    = _score(PROMPT_FAITHFULNESS.format(context=context, answer=answer))
    relevancy       = _score(PROMPT_RELEVANCY.format(question=question, answer=answer))
    ctx_quality     = _score(PROMPT_CONTEXT_QUALITY.format(question=question, context=context))
    # Les 2 métriques avec ground truth
    ctx_recall      = _score(PROMPT_CONTEXT_RECALL.format(ground_truth=ground_truth, context=context))
    answer_correct  = _score(PROMPT_ANSWER_CORRECTNESS.format(ground_truth=ground_truth, answer=answer))

    print(f"  Faithfulness       : {faithfulness:.2f}")
    print(f"  Answer Relevancy   : {relevancy:.2f}")
    print(f"  Context Quality    : {ctx_quality:.2f}")
    print(f"  Context Recall     : {ctx_recall:.2f}")
    print(f"  Answer Correctness : {answer_correct:.2f}")

    return {
        "id": entry["id"],
        "niveau": entry["niveau"],
        "type": entry["type"],
        "question": question,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_quality": ctx_quality,
        "context_recall": ctx_recall,
        "answer_correctness": answer_correct,
    }


def print_results(results: list[dict]):
    print(f"\n{'='*90}")
    print("RÉSULTATS COMPLETS")
    print(f"{'='*90}")
    header = f"{'ID':<15} {'Niv':>4} {'Faith.':>7} {'Relev.':>7} {'Ctx.Q':>7} {'Recall':>7} {'Correct.':>9}"
    print(header)
    print("-"*90)

    for r in results:
        print(f"{r['id']:<15} {r['niveau']:>4} {r['faithfulness']:>7.2f} {r['answer_relevancy']:>7.2f} "
              f"{r['context_quality']:>7.2f} {r['context_recall']:>7.2f} {r['answer_correctness']:>9.2f}")

    print("-"*90)
    # Moyennes globales
    def avg(key): return sum(r[key] for r in results) / len(results)
    print(f"{'MOYENNE GLOBALE':<15} {'':>4} {avg('faithfulness'):>7.2f} {avg('answer_relevancy'):>7.2f} "
          f"{avg('context_quality'):>7.2f} {avg('context_recall'):>7.2f} {avg('answer_correctness'):>9.2f}")

    # Moyennes par niveau
    print(f"\n{'─'*40}")
    print("Moyennes par niveau :")
    for niveau in [1, 2, 3]:
        sub = [r for r in results if r["niveau"] == niveau]
        if not sub:
            continue
        print(f"  Niveau {niveau} ({len(sub)} questions) : "
              f"faith={sum(r['faithfulness'] for r in sub)/len(sub):.2f}  "
              f"relev={sum(r['answer_relevancy'] for r in sub)/len(sub):.2f}  "
              f"recall={sum(r['context_recall'] for r in sub)/len(sub):.2f}  "
              f"correct={sum(r['answer_correctness'] for r in sub)/len(sub):.2f}")


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8")) 

    print("=== Évaluation RAG ===")
    print(f"LLM          : {GEN_MODEL}")
    print(f"Embedding    : {EMBED_MODEL}")
    print(f"Dataset      : {DATASET_PATH.name} ({len(dataset)} questions)")
    print(f"Métriques    : Faithfulness, Answer Relevancy, Context Quality, Context Recall, Answer Correctness")

    output_path = DATASET_PATH.parent / "results.json"
    results = []
    try:
        for entry in dataset:
            result = evaluate_question(entry)
            if result:
                results.append(result)
                # Sauvegarde après chaque question — Ctrl+C ne perd rien
                output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        print("\n\nInterrompu — résultats partiels sauvegardés.")

    if results:
        print_results(results)
        print(f"\nRésultats sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()
