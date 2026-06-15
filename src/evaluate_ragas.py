# Script d'évaluation RAGAS du pipeline RAG agentic.
# Charge data/agentic_results.json, construit un dataset RAGAS et calcule 5 métriques clés.
# Tout tourne en local via Ollama — aucun appel à OpenAI.

import json
import sys
import pandas as pd
from pathlib import Path

# --- Imports RAGAS (nécessite : pip install ragas) ---
import asyncio
import inspect
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,        # L'answer est-elle fondée sur le contexte récupéré ?
    AnswerRelevancy,     # L'answer répond-elle à la question posée ?
    ContextPrecision,    # Les chunks récupérés sont-ils pertinents pour répondre ?
    ContextRecall,       # Le contexte couvre-t-il l'information de la réponse attendue ?
    AnswerCorrectness,   # L'answer est-elle correcte par rapport à la réponse attendue ?
)
from ragas.llms import llm_factory               # Crée un InstructorLLM requis par ragas.metrics.collections
from ragas.embeddings import embedding_factory   # Crée un BaseRagasEmbedding requis par ragas.metrics.collections

# --- Import OpenAI (client async OpenAI-compatible pour pointer vers Ollama) ---
from openai import AsyncOpenAI  # ascore() appelle agenerate() — nécessite un client async

# =============================================================================
# CONFIGURATION — modifier ici selon l'environnement
# =============================================================================
RESULTS_PATH  = Path("data/agentic_results.json")   # Résultats du pipeline agentic
QUESTIONS_PATH = Path("data/questions.json")         # Dataset complet (métadonnées)
OUTPUT_CSV    = Path("data/ragas_evaluation.csv")    # Fichier de sortie des scores

EVAL_LLM     = "gemma4:12b"   # Modèle Ollama utilisé pour les jugements LLM de RAGAS
EMBED_MODEL  = "bge-m3"       # Modèle d'embedding pour la métrique AnswerRelevancy
# =============================================================================


def load_data() -> tuple[list[dict], dict[str, dict]]:
    """Charge les résultats agentic et le dataset de questions (pour les métadonnées)."""
    # Vérification de l'existence des fichiers avant ouverture
    if not RESULTS_PATH.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {RESULTS_PATH}")
    if not QUESTIONS_PATH.exists():
        sys.exit(f"[ERREUR] Fichier introuvable : {QUESTIONS_PATH}")

    # Lecture du fichier de résultats produit par run_agentic_all.py
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    # Lecture du dataset de questions pour récupérer niveau / difficulté / source
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    # Indexation des questions par id pour un accès rapide
    meta_by_id = {q["id"]: q for q in questions}

    return results, meta_by_id


def build_dataset(results: list[dict]) -> tuple[EvaluationDataset, list[dict]]:
    """Construit un EvaluationDataset RAGAS à partir des résultats agentic."""
    samples = []       # Liste de SingleTurnSample pour RAGAS
    row_meta = []      # Métadonnées conservées pour l'export CSV final

    for r in results:
        # Extraction des chunks récupérés par le pipeline — chaque chunk devient une string de contexte
        contexts = [
            f"[Source : {c['source']} — p.{c['page']}]\n{c['content']}"
            for c in r.get("chunks", [])
            if c.get("content", "").strip()  # Filtre les chunks vides
        ]

        # On ignore les lignes sans contexte (retrieval échoué ou résultat corrompu)
        if not contexts:
            print(f"  [SKIP] {r['id']} — aucun chunk disponible")
            continue

        # Création d'un échantillon RAGAS avec tous les champs nécessaires aux 5 métriques
        sample = SingleTurnSample(
            user_input=r["question"],             # Question posée à l'agent
            response=r["reponse_llm"],            # Réponse générée par le LLM
            retrieved_contexts=contexts,           # Liste de chunks récupérés (strings)
            reference=r["reponse_attendue"],       # Réponse attendue (ground truth)
        )
        samples.append(sample)

        # Conservation de l'id et de la question pour enrichir le CSV de sortie
        row_meta.append({"id": r["id"], "question": r["question"]})

    # Création du dataset RAGAS à partir de la liste de samples
    dataset = EvaluationDataset(samples=samples)
    return dataset, row_meta


def build_metrics(llm, embeddings) -> list:
    """Instancie les 5 métriques RAGAS configurées avec un InstructorLLM et un BaseRagasEmbedding."""
    return [
        Faithfulness(llm=llm),                            # Basé uniquement sur le contexte
        AnswerRelevancy(llm=llm, embeddings=embeddings),  # Utilise les embeddings
        ContextPrecision(llm=llm),                        # Juge la précision des chunks
        ContextRecall(llm=llm),                           # Couvreture vs ground truth
        AnswerCorrectness(llm=llm, embeddings=embeddings), # Score global de correction
    ]


def _build_kwargs(method, sample: SingleTurnSample) -> dict:
    """Mappe les champs de SingleTurnSample aux paramètres attendus par ascore() via inspection."""
    field_map = {
        "user_input":          sample.user_input,
        "response":            sample.response,
        "retrieved_contexts":  sample.retrieved_contexts,
        "reference":           sample.reference,
    }
    sig = inspect.signature(method)
    # Garde uniquement les paramètres présents dans field_map (ignore *args, **kwargs)
    return {
        name: field_map[name]
        for name, param in sig.parameters.items()
        if name in field_map and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


async def _score_async(dataset: EvaluationDataset, metrics: list) -> pd.DataFrame:
    """Score chaque sample individuellement — contourne evaluate() incompatible avec les collections metrics."""
    rows = []
    total = len(dataset.samples)
    for i, sample in enumerate(dataset.samples, 1):
        # En-tête de la question avec numéro et texte tronqué
        question_preview = sample.user_input[:120] + "…" if len(sample.user_input) > 120 else sample.user_input
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{total}] {question_preview}")
        print(f"{'─' * 60}")

        row = {}
        for metric in metrics:
            try:
                if hasattr(metric, "ascore"):
                    # ascore() attend les champs séparés, pas un SingleTurnSample
                    kwargs = _build_kwargs(metric.ascore, sample)
                    result = metric.ascore(**kwargs)
                    score = await result if asyncio.iscoroutine(result) else result
                    # MetricResult expose .value — on extrait le float brut pour le CSV
                    row[metric.name] = score.value if hasattr(score, "value") else score
                elif hasattr(metric, "score"):
                    kwargs = _build_kwargs(metric.score, sample)
                    result = metric.score(**kwargs)
                    score = await result if asyncio.iscoroutine(result) else result
                    row[metric.name] = score.value if hasattr(score, "value") else score
                else:
                    row[metric.name] = None
            except Exception as e:
                row[metric.name] = None
                print(f"  [WARN] {metric.name} : {e}")

            # Affichage du score de la métrique avec barre visuelle
            val = row[metric.name]
            if val is None:
                print(f"  {metric.name:<25}  N/A")
            else:
                bar = "█" * int(val * 20)
                print(f"  {metric.name:<25}  {val:.3f}  {bar}")

        rows.append(row)
    return pd.DataFrame(rows)


def _run_scoring(dataset: EvaluationDataset, metrics: list) -> pd.DataFrame:
    """Lance _score_async en gérant les deux cas : script Python et Jupyter/Colab (event loop déjà active)."""
    try:
        return asyncio.run(_score_async(dataset, metrics))       # Cas standard (script Python)
    except RuntimeError:
        # Jupyter/Colab : event loop déjà active — nest_asyncio permet l'imbrication
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_score_async(dataset, metrics))


def print_summary(df: pd.DataFrame) -> None:
    """Affiche un récapitulatif par question puis les scores moyens dans le terminal."""
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]
    # Abréviations pour tenir dans la ligne
    abbr = {
        "faithfulness":       "faith",
        "answer_relevancy":   "ans_rel",
        "context_precision":  "ctx_prec",
        "context_recall":     "ctx_rec",
        "answer_correctness": "ans_corr",
    }
    present = [c for c in metric_cols if c in df.columns]

    # --- Tableau par question ---
    print("\n" + "=" * 85)
    print("  RÉSULTATS PAR QUESTION")
    print("=" * 85)

    # En-tête du tableau
    header = f"  {'ID':<14}" + "".join(f"  {abbr[c]:>8}" for c in present)
    print(header)
    print("  " + "─" * 83)

    for _, row in df.iterrows():
        # Question tronquée à 14 chars pour tenir dans la colonne ID
        qid = str(row.get("id", ""))[:14]
        scores = ""
        for c in present:
            val = row[c]
            scores += f"  {val:>8.3f}" if pd.notna(val) else f"  {'N/A':>8}"
        print(f"  {qid:<14}{scores}")

    # --- Moyennes globales ---
    print("\n" + "=" * 55)
    print("  SCORES MOYENS")
    print("=" * 55)

    for col in present:
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            print(f"  {col:<25} N/A")
            continue
        bar = "█" * int(mean_val * 20)
        print(f"  {col:<25} {mean_val:.3f}  {bar}")

    print(f"\n  Questions évaluées : {len(df)}")

    # Ventilation par niveau si la colonne est présente dans le CSV
    if "niveau" in df.columns:
        print("\n  --- Par niveau ---")
        for niveau, grp in df.groupby("niveau"):
            mean_corr = grp["answer_correctness"].mean() if "answer_correctness" in grp else float("nan")
            print(f"  Niveau {niveau} ({len(grp)} qs) → answer_correctness = {mean_corr:.3f}")

    print("=" * 55)


def main() -> None:
    # Chargement des données sources
    print("[1/5] Chargement des données...")
    results, meta_by_id = load_data()
    print(f"      {len(results)} résultats chargés depuis {RESULTS_PATH}")

    # Construction du dataset RAGAS
    print("[2/5] Construction du dataset RAGAS...")
    dataset, row_meta = build_dataset(results)
    print(f"      {len(dataset)} samples prêts pour l'évaluation")

    if len(dataset) == 0:
        sys.exit("[ERREUR] Aucun sample valide — vérifiez agentic_results.json")

    # Initialisation des modèles locaux Ollama
    print(f"[3/5] Initialisation des modèles Ollama ({EVAL_LLM}, {EMBED_MODEL})...")

    # Ollama expose une API compatible OpenAI sur le port 11434 — llm_factory en a besoin
    # pour produire un InstructorLLM accepté par ragas.metrics.collections
    # AsyncOpenAI requis car ascore() appelle agenerate() en interne
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Valeur fictive : Ollama ne vérifie pas la clé
    )
    llm           = llm_factory(EVAL_LLM, client=ollama_client)
    # Ollama expose /v1/embeddings compatible OpenAI — on passe "openai" comme provider
    embeddings    = embedding_factory("openai", model=EMBED_MODEL, client=ollama_client, interface="modern")

    # Instanciation des métriques avec les modèles locaux
    metrics = build_metrics(llm, embeddings)
    print(f"      {len(metrics)} métriques configurées : {[m.name for m in metrics]}")

    # Lancement de l'évaluation — peut prendre plusieurs minutes selon le nombre de questions
    print("[4/5] Évaluation en cours (peut durer plusieurs minutes)...")
    df_scores = _run_scoring(dataset, metrics)  # Scoring async sample par sample

    # Export CSV enrichi avec id + métadonnées
    print("[5/5] Export des résultats...")
    df_meta   = pd.DataFrame(row_meta)                         # DataFrame des identifiants

    df_out = pd.concat([df_meta.reset_index(drop=True),
                        df_scores.reset_index(drop=True)], axis=1)

    # Ajout des métadonnées questions (niveau, source, difficulté) pour l'analyse
    for col in ["niveau", "source", "difficulte_rag"]:
        df_out[col] = df_out["id"].map(
            lambda qid, c=col: meta_by_id.get(qid, {}).get(c, "")
        )

    # Sauvegarde du CSV final
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"      Scores sauvegardés → {OUTPUT_CSV}")

    # Affichage du résumé dans le terminal
    print_summary(df_out)


if __name__ == "__main__":
    main()
