# Script d'évaluation RAGAS du pipeline RAG agentic.
# Charge data/agentic_results.json, construit un dataset RAGAS et calcule 5 métriques clés.
# Tout tourne en local via Ollama — aucun appel à OpenAI.

import json
import sys
import pandas as pd
from pathlib import Path

# --- Imports RAGAS (nécessite : pip install ragas) ---
import asyncio
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

# --- Import OpenAI (client OpenAI-compatible pour pointer vers Ollama) ---
from openai import OpenAI

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


def _call_metric(metric, sample) -> float | None:
    """Appelle la méthode de scoring disponible selon la version de RAGAS installée."""
    # RAGAS v0.2+ collections : méthode score() synchrone
    if hasattr(metric, "score"):
        return metric.score(sample)
    # RAGAS v0.1 : méthode ascore() asynchrone (ne devrait pas arriver ici mais par sécurité)
    return None


async def _score_async(dataset: EvaluationDataset, metrics: list) -> pd.DataFrame:
    """Score chaque sample individuellement — contourne evaluate() incompatible avec les collections metrics."""
    # Affiche les méthodes disponibles au premier lancement pour diagnostic
    if dataset.samples:
        available = [m for m in dir(metrics[0]) if not m.startswith("_") and "score" in m.lower()]
        print(f"  [INFO] méthodes scoring détectées sur {metrics[0].__class__.__name__} : {available}")

    rows = []
    total = len(dataset.samples)
    for i, sample in enumerate(dataset.samples, 1):
        row = {}
        for metric in metrics:
            try:
                if hasattr(metric, "single_turn_ascore"):       # RAGAS ≥ 0.2 async
                    row[metric.name] = await metric.single_turn_ascore(sample)
                elif hasattr(metric, "ascore"):                  # RAGAS 0.1 async
                    row[metric.name] = await metric.ascore(sample)
                elif hasattr(metric, "score"):                   # RAGAS 0.2 collections sync
                    result = metric.score(sample)
                    # score() peut retourner une coroutine selon la version
                    row[metric.name] = await result if asyncio.iscoroutine(result) else result
                else:
                    row[metric.name] = None
            except Exception as e:
                row[metric.name] = None
                if i == 1:  # Affiche l'erreur seulement au premier sample pour ne pas spammer
                    print(f"  [WARN] {metric.name} : {e}")
        print(f"  [{i}/{total}] scoré")
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
    """Affiche un récapitulatif des scores moyens dans le terminal."""
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]

    print("\n" + "=" * 55)
    print("  RÉSULTATS RAGAS — SCORES MOYENS")
    print("=" * 55)

    # Affichage des scores globaux sur toutes les questions évaluées
    for col in metric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            if pd.isna(mean_val):  # Scores NaN = metric n'a pas pu être calculée
                print(f"  {col:<25} N/A")
                continue
            bar = "█" * int(mean_val * 20)  # Barre de progression visuelle (20 = 100%)
            print(f"  {col:<25} {mean_val:.3f}  {bar}")

    print(f"\n  Questions évaluées : {len(df)}")

    # Ventilation par niveau si la colonne est présente dans le CSV
    if "niveau" in df.columns:
        print("\n  --- Par niveau ---")
        for niveau, grp in df.groupby("niveau"):
            # Score moyen par niveau (L1 facile / L2 intermédiaire)
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
    ollama_client = OpenAI(
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
