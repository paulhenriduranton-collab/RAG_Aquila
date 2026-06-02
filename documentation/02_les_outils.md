# 02 — Les outils utilisés

## Vue d'ensemble

Le projet utilise 6 outils principaux. Voici leur rôle exact.

---

## 1. Ollama

**Ce que c'est :** Un logiciel qui permet de faire tourner des modèles d'IA directement sur ton ordinateur, sans connexion internet.

**Ce qu'il fait dans ce projet :** Il héberge deux modèles :
- `nomic-embed-text` pour transformer du texte en vecteurs
- `gemma:latest` pour générer les réponses

**Analogie :** C'est comme un serveur de restaurant — il ne cuisine pas lui-même, mais il gère les commandes et les envoie aux bons cuisiniers (les modèles).

---

## 2. nomic-embed-text

**Ce que c'est :** Un modèle d'IA spécialisé dans la transformation de texte en vecteurs.

**Ce qu'il fait dans ce projet :** À chaque morceau de texte, il attribue une liste de 768 nombres. Ces nombres représentent le *sens* du texte d'une façon mathématique.

**Exemple concret :**
- "théorème de Pythagore" → `[0.12, -0.87, 0.34, ...]` (768 nombres)
- "triangle rectangle" → `[0.11, -0.85, 0.36, ...]` (nombres très proches = sens proche)
- "recette de gâteau" → `[-0.92, 0.14, -0.67, ...]` (nombres très différents = sens différent)

**Pourquoi 768 nombres ?** C'est la "taille" de ce modèle. D'autres modèles utilisent 512, 1024, ou 1536 dimensions. Plus il y en a, plus la représentation est précise.

---

## 3. gemma:latest

**Ce que c'est :** Un modèle de langage (LLM) créé par Google, qui tourne en local via Ollama.

**Ce qu'il fait dans ce projet :** Il reçoit le prompt (ta question + les passages trouvés + les instructions) et génère une réponse rédigée en français.

**Comparaison des tailles disponibles :**
| Modèle | Taille | Qualité | Vitesse |
|---|---|---|---|
| gemma:2b | 1.7 GB | Faible — hallucine souvent | Rapide |
| gemma:latest | 5.0 GB | Bonne | Moyen |
| gemma4:latest | 9.6 GB | Très bonne | Lent |

On utilise `gemma:latest` dans ce projet — bon compromis qualité/vitesse.

---

## 4. ChromaDB

**Ce que c'est :** Une base de données spécialisée dans le stockage et la recherche de vecteurs.

**Ce qu'il fait dans ce projet :** Il stocke les 768 nombres de chaque morceau de texte dans un fichier SQLite sur ton disque (`vector_db/chroma.sqlite3`). Quand tu poses une question, il calcule quel vecteur est le plus proche du vecteur de ta question.

**La différence avec une base de données classique :**

Une base classique (Excel, MySQL) cherche des correspondances exactes :
- Tu cherches "Banach" → il trouve les lignes qui contiennent exactement le mot "Banach"

Chroma cherche par proximité de sens :
- Tu cherches "espace complet" → il trouve les passages sur "Banach", "Cauchy", "convergence" même si le mot exact n'y est pas

**Le fichier `chroma.sqlite3` :** C'est le fichier qui contient toute la base. SQLite est un format de base de données très courant (une application mobile sur deux l'utilise). C'est un seul fichier, pas besoin d'un serveur séparé.

---

## 5. LangChain

**Ce que c'est :** Une librairie Python qui sert de "colle" entre tous les outils.

**Ce qu'il fait dans ce projet :** Sans LangChain, il faudrait écrire du code complexe pour connecter Ollama, Chroma, les loaders de PDF, etc. LangChain fournit des blocs prêts à l'emploi :
- `PyPDFLoader` → lit un PDF
- `RecursiveCharacterTextSplitter` → découpe le texte
- `OllamaEmbeddings` → appelle nomic-embed-text
- `Chroma` → gère la base vectorielle
- `OllamaLLM` → appelle gemma

**Analogie :** LangChain c'est comme une boîte à outils Ikea — les pièces sont déjà fabriquées, tu n'as plus qu'à les assembler.

---

## 6. Streamlit

**Ce que c'est :** Une librairie Python qui crée des interfaces web très simplement.

**Ce qu'il fait dans ce projet :** Il transforme le fichier `app.py` (30 lignes de Python) en une page web avec un champ texte et un bouton. Pas besoin de connaître HTML, CSS ou JavaScript.

**Comment ça marche :** Tu lances `streamlit run src/app.py` dans le terminal, et Streamlit ouvre automatiquement `http://localhost:8501` dans ton navigateur.
