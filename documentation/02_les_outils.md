# 02 — Les outils utilisés

## Vue d'ensemble

Le projet utilise 7 outils principaux. Voici leur rôle exact.

---

## 1. Ollama

**Ce que c'est :** Un logiciel qui permet de faire tourner des modèles d'IA directement sur ton ordinateur, sans connexion internet.

**Ce qu'il fait dans ce projet :** Il héberge deux modèles :
- `bge-m3` pour transformer du texte en vecteurs (embeddings)
- `gemma2:2b` pour générer les réponses

**Analogie :** C'est comme un serveur de restaurant — il ne cuisine pas lui-même, mais il gère les commandes et les envoie aux bons cuisiniers (les modèles).

---

## 2. bge-m3 (modèle d'embedding)

**Ce que c'est :** Un modèle d'IA spécialisé dans la transformation de texte en vecteurs, développé par BAAI (Beijing Academy of AI).

**Ce qu'il fait dans ce projet :** À chaque morceau de texte, il attribue une liste de 1024 nombres. Ces nombres représentent le *sens* du texte d'une façon mathématique.

**Pourquoi bge-m3 et pas un autre ?**

| Modèle | Langues | Fenêtre contexte | Adapté aux maths FR |
|---|---|---|---|
| nomic-embed-text | Anglais surtout | 8192 tokens | Non |
| mxbai-embed-large | Multilingue | **512 tokens** | Partiel (trop court) |
| **bge-m3** | **100+ langues dont FR** | **8192 tokens** | **Oui** |

`bge-m3` est le seul qui combine : bonne compréhension du français, grande fenêtre de contexte (pour des chunks de 1000 caractères), et entraînement sur des textes scientifiques multilingues.

**Exemple concret :**
- "estimateur sans biais" → `[0.23, -0.71, 0.44, ...]` (1024 nombres)
- "unbiased estimator" → `[0.22, -0.70, 0.43, ...]` (très proches → même concept)
- "recette de gâteau" → `[-0.81, 0.34, -0.52, ...]` (très différents → autre sujet)

**Limite importante :** bge-m3 comprend le *vocabulaire* mathématique ("différentielle", "estimateur", "convergence") mais pas les *formules* brutes (`∫₀¹ f(x)dx`, `d²f/dx²`). C'est pour ça qu'on combine le sémantique avec BM25.

---

## 3. BM25 (recherche lexicale)

**Ce que c'est :** Un algorithme de recherche par mots-clés, utilisé dans des moteurs de recherche comme Elasticsearch. BM25 = *Best Match 25*.

**Ce qu'il fait dans ce projet :** En parallèle de la recherche sémantique, BM25 cherche les chunks qui contiennent exactement les mots de la question — utile pour les termes mathématiques précis et les formules.

**La différence avec la recherche sémantique :**

| Recherche sémantique (bge-m3) | Recherche lexicale (BM25) |
|---|---|
| Cherche par *sens* | Cherche par *mots exacts* |
| "espace complet" trouve "Banach" | "différentielle" trouve "différentielle" |
| Bonne sur les concepts | Bonne sur les termes techniques |
| Peut se tromper de matière | Ne se trompe pas de vocabulaire |

**En pratique :** La question "Différentielle d'ordre 2" est reconnue par BM25 dans le bon poly (`calcul_diff.pdf`) même si la recherche sémantique retourne des résultats d'un autre cours.

---

## 4. RRF — Reciprocal Rank Fusion

**Ce que c'est :** Un algorithme qui fusionne les résultats de deux listes classées (sémantique + BM25) en un seul classement.

**La formule :**
```
score_RRF(chunk) = 1/(60 + rang_sémantique) + 1/(60 + rang_BM25)
```

**Pourquoi pas juste additionner les scores bruts ?**

Parce que les scores sémantiques (entre 0 et 1) et les scores BM25 (entre 0 et 20+) n'ont pas la même échelle. Un simple ajout favorise toujours l'une des deux méthodes. RRF utilise les *rangs* (position dans la liste), pas les valeurs brutes — ce qui rend la fusion équitable.

**Exemple :**
- `calcul_diff.pdf` est rang #1 en BM25 → score RRF = 1/61 = 0.0164
- `stats.pdf` est rang #1 en sémantique → score RRF = 1/61 = 0.0164
- Si `calcul_diff.pdf` apparaît aussi dans les deux listes → ses contributions s'additionnent → il remonte en tête

---

## 5. pymupdf4llm (extraction PDF)

**Ce que c'est :** Une extension de PyMuPDF spécialement conçue pour produire un Markdown propre depuis les PDFs, optimisée pour les LLMs.

**Pourquoi pas PyMuPDF brut ?**

PyMuPDF brut colle les mots avec les symboles mathématiques :
```
# PyMuPDF brut (avant)
"SoientX⊂R d un ouvert etf:X→R m d2fdx2"

# pymupdf4llm (maintenant)
"Soient X ⊂ R^d un ouvert et f : X → R^m"
```

pymupdf4llm préserve aussi les titres de sections (`## Chapitre 3`), les tableaux en Markdown, et les listes à puces — ce qui améliore la qualité du découpage en chunks.

**Limite :** Il ne reconnaît pas les formules LaTeX. `∫₀¹ f(x)dx` reste du texte brut, pas du LaTeX. Pour ça il faudrait Nougat (Meta) qui nécessite Python ≤ 3.13.

---

## 6. gemma2:2b

**Ce que c'est :** Un modèle de langage (LLM) créé par Google, version 2 milliards de paramètres, tournant en local via Ollama.

**Ce qu'il fait dans ce projet :** Il reçoit le prompt (ta question + les 5 passages trouvés + les instructions strictes) et génère une réponse rédigée en français, uniquement à partir du contexte fourni.

**Comparaison des modèles de génération :**
| Modèle | Taille | Qualité | Vitesse |
|---|---|---|---|
| gemma2:2b | 1.6 GB | Correcte | Très rapide |
| gemma:latest | 5.0 GB | Bonne | Moyen |
| gemma2:9b | 5.5 GB | Très bonne | Lent |

On utilise `gemma2:2b` — suffisant pour extraire et reformuler des informations depuis un contexte fourni.

---

## 7. ChromaDB

**Ce que c'est :** Une base de données spécialisée dans le stockage et la recherche de vecteurs.

**Ce qu'il fait dans ce projet :** Il stocke les 1024 nombres de chaque chunk dans `vector_db/chroma.sqlite3`. Quand tu poses une question, il calcule les 20 vecteurs les plus proches du vecteur de ta question.

**La différence avec une base de données classique :**

Une base classique cherche des correspondances exactes :
- Tu cherches "Banach" → il trouve les lignes qui contiennent exactement "Banach"

ChromaDB cherche par proximité de sens :
- Tu cherches "espace complet" → il trouve les passages sur "Banach", "Cauchy", "convergence" même si le mot exact n'y est pas

---

## 8. LangChain

**Ce que c'est :** Une librairie Python qui sert de "colle" entre tous les outils.

**Ce qu'il fait dans ce projet :** Fournit des blocs prêts à l'emploi :
- `RecursiveCharacterTextSplitter` → découpe le texte
- `OllamaEmbeddings` → appelle bge-m3
- `Chroma` → gère la base vectorielle
- `OllamaLLM` → appelle gemma2:2b

**Analogie :** LangChain c'est comme une boîte à outils — les pièces sont déjà fabriquées, tu n'as plus qu'à les assembler.
