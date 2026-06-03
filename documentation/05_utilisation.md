# 05 — Guide d'utilisation

## Prérequis

Avant de commencer, vérifie que tu as :
- **Python** installé (version 3.10 à 3.13 — pas 3.14, incompatibilité avec certaines dépendances de marker)
- **Ollama** installé et lancé (visible dans la barre des tâches)
- Les deux modèles téléchargés :
  ```powershell
  ollama pull bge-m3
  ollama pull gemma2:2b
  ```

---

## Installation (à faire une seule fois)

### 1. Créer l'environnement virtuel

Un environnement virtuel est un dossier qui contient les librairies Python du projet, séparées du reste de ton ordinateur.

```powershell
python -m venv venv
venv\Scripts\activate
```

Quand l'environnement est actif, tu vois `(venv)` au début de chaque ligne du terminal.

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
pip install pymupdf4llm
```

`pymupdf4llm` est installé séparément car il n'est pas dans le `requirements.txt` d'origine — c'est l'extracteur PDF amélioré pour les polys de maths.

---

## Utilisation normale

### Étape 1 — Ajouter tes documents

Copie tes fichiers (PDF, Word ou TXT) dans le dossier `documents/`.

### Étape 2 — Indexer les documents

```powershell
python src/ingest.py
```

Tu verras s'afficher :
```
Chargement des documents...
  ✓ analyse_fonctionnelle.pdf (1 doc(s))
  ✓ calcul_diff.pdf (1 doc(s))
  ✓ Poly-L3-StatMath (1).pdf (1 doc(s))

3 document(s) chargé(s).
611 chunk(s) créé(s).
Lot 1 / 13 (50 chunks)...
...
Index créé dans vector_db/.
```

**Durée :** 10 à 30 minutes selon ta machine. À ne refaire que si tu ajoutes de nouveaux documents.

### Étape 3 — Poser une question

```powershell
python src/ask.py
```

Le terminal affiche `Question :`. Tape ta question et appuie sur Entrée.

Tu verras s'afficher le détail de la recherche :

```
[DB] 611 chunks dans la base

[Sémantique] Recherche des 20 plus proches voisins...
[Sémantique] Top 5 :
  #1  score=0.721  calcul_diff.pdf  p.5
        ↳ Calcul Différentiel I. Différentielles d'ordre supérieur...
  ...

[BM25] Top 5 résultats lexicaux :
  #1  bm25=16.03  calcul_diff.pdf  p.5
        ↳ Calcul Différentiel I. Différentielles d'ordre supérieur...
  ...

[Fusion] Top 5 après fusion sémantique + BM25 :
  rrf=0.0328  calcul_diff.pdf  p.5
  ...

Réponse :
La différentielle d'ordre 2 est définie comme...
```

Utilise **Ctrl+C** pour quitter.

---

## Si tu ajoutes de nouveaux documents

1. Copie les nouveaux fichiers dans `documents/`
2. Supprime l'ancienne base : `Remove-Item -Recurse -Force vector_db`
3. Relance `python src/ingest.py`
4. Relance `python src/ask.py`

**Pourquoi supprimer `vector_db/` ?** La base contient les vecteurs produits par bge-m3. Si tu ajoutes simplement des documents sans vider la base, les anciens vecteurs (potentiellement d'un ancien modèle d'embedding) coexistent avec les nouveaux — ce qui peut causer des incohérences.

---

## Vérifier qu'Ollama fonctionne

```powershell
ollama list
```

Tu dois voir `bge-m3:latest` et `gemma2:2b` dans la liste. Si ce n'est pas le cas :
```powershell
ollama pull bge-m3
ollama pull gemma2:2b
```

---

## Lire les logs de recherche

À chaque question, le système affiche trois sections :

| Section | Ce que ça montre |
|---|---|
| `[Sémantique] Top 5` | Les 5 meilleurs chunks par similarité vectorielle (bge-m3) |
| `[BM25] Top 5` | Les 5 meilleurs chunks par correspondance de mots-clés |
| `[Fusion] Top 5` | Les 5 chunks finaux après RRF, envoyés au LLM |

Si la recherche sémantique retourne les mauvais documents mais que le BM25 est correct, c'est normal — la fusion RRF donnera plus de poids aux chunks présents dans les deux listes.
