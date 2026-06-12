# Script d'évaluation RAGAS du pipeline RAG agentic.
# Charge data/agentic_results.json, construit un dataset RAGAS et calcule 5 métriques clés.
# Tout tourne en local via Ollama — aucun appel à OpenAI.

import json
import sys
import pandas as pd
from pathlib import Path

# --- Imports RAGAS (nécessite : pip install ragas) ---
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,        # L'answer est-elle fondée sur le contexte récupéré ?
    AnswerRelevancy,     # L'answer répond-elle à la question posée ?
    ContextPrecision,    # Les chunks récupérés sont-ils pertinents pour répondre ?
    ContextRecall,       # Le contexte couvre-t-il l'information de la réponse attendue ?
    AnswerCorrectness,   # L'answer est-elle correcte par rapport à la réponse attendue ?
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# --- Imports LangChain Ollama ---
from langchain_ollama import ChatOllama, OllamaEmbeddings

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


def build_metrics(llm_wrapper, embed_wrapper) -> list:
    """Instancie les 5 métriques RAGAS configurées avec le LLM et les embeddings locaux."""
    return [
        Faithfulness(llm=llm_wrapper),                              # Basé uniquement sur le contexte
        AnswerRelevancy(llm=llm_wrapper, embeddings=embed_wrapper), # Utilise les embeddings
        ContextPrecision(llm=llm_wrapper),                          # Juge la précision des chunks
        ContextRecall(llm=llm_wrapper),                             # Couvreture vs ground truth
        AnswerCorrectness(llm=llm_wrapper),                         # Score global de correction
    ]


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
    llm_wrapper   = LangchainLLMWrapper(ChatOllama(model=EVAL_LLM, temperature=0))
    embed_wrapper = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

    # Instanciation des métriques avec les modèles locaux
    metrics = build_metrics(llm_wrapper, embed_wrapper)
    print(f"      {len(metrics)} métriques configurées : {[m.name for m in metrics]}")

    # Lancement de l'évaluation — peut prendre plusieurs minutes selon le nombre de questions
    print("[4/5] Évaluation en cours (peut durer plusieurs minutes)...")
    result = evaluate(dataset=dataset, metrics=metrics)

    # Export CSV enrichi avec id + métadonnées
    print("[5/5] Export des résultats...")
    df_scores = result.to_pandas()                             # DataFrame des scores RAGAS
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
