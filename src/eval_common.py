# eval_common.py
# Utilitaires partagés entre run_agentic_all.py (exécution du pipeline, une seule fois,
# qui sauvegarde tout dans agentic_results.json) et evaluate_components.py (scoring RAGAS,
# rejoué à partir de ce fichier sans refaire tourner retrieval/génération) :
# mapping des sources pour la ground truth du router, conversion chunks → contextes RAGAS.

from agent import _available_sources

# Mapping source (questions.json) → fichiers PDF — rempli au démarrage par init_source_map()
SOURCE_MAP: dict[str, list[str]] = {}


def init_source_map():
    """Détecte les PDF dans documents/ et construit le mapping source → fichier(s)."""
    global SOURCE_MAP
    available = _available_sources()
    SOURCE_MAP.clear()
    for name in available:
        upper = name.upper()
        if "ENS" in upper:
            SOURCE_MAP.setdefault("ENS", []).append(name)
        else:
            SOURCE_MAP.setdefault("Sorbonne", []).append(name)
    # ENS+Sorbonne = pas de filtre, toutes les sources
    SOURCE_MAP["ENS+Sorbonne"] = available
    print(f"[Config] Sources détectées : {SOURCE_MAP}")


def expected_sources(meta: dict) -> list[str] | None:
    """Fichiers PDF attendus pour cette question. None = toutes les sources."""
    src = meta.get("source", "")
    if "+" in src:
        return None
    return SOURCE_MAP.get(src)


def expected_difficulty(meta: dict) -> int:
    """Difficulté attendue du router (niveau 1→factuel, 2→synthèse, 3→complexe)."""
    return meta.get("niveau", 2)


def chunks_to_contexts(chunks: list[dict]) -> list[str]:
    """Convertit des chunks sérialisés (schéma agentic_results.json : source/page/content)
    en liste de strings pour RAGAS."""
    return [
        f"[Source : {c.get('source', '?')} — p.{c.get('page', '?')}]\n{c['content']}"
        for c in chunks
        if c.get("content", "").strip()
    ]
