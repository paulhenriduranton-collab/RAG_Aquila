import os
import pickle
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import ftfy          # répare les encodages cassés dans les textes extraits de PDF
import pymupdf4llm  # convertit les PDF en Markdown structuré (meilleur que l'extraction brute PyMuPDF)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # base de données vectorielle locale
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # génère les embeddings et fait le split agentic via Ollama
from langchain_community.document_loaders import TextLoader, Docx2txtLoader  # loaders pour .txt et .docx
from langchain_core.documents import Document  # objet LangChain : texte + métadonnées

# Chemins calculés dynamiquement depuis l'emplacement de ce fichier
BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
DOCUMENTS_DIR = BASE_DIR / "documents"             # dossier où déposer les fichiers à indexer
VECTOR_DB_DIR = Path("C:/vector_db_aquila")        # hors OneDrive — SQLite corrompu par la synchro cloud
EMBED_MODEL = "bge-m3"  # modèle d'embedding multilingue — doit être le même que dans ask.py
SPLIT_MODEL = "gemma2:2b"  # LLM pour le split agentic — gemma2:2b suffisant pour insérer des marqueurs

# Seuils du pipeline
MAX_CHUNK_SIZE = 2000         # au-delà, un chunk est redécoupé par le fallback déterministe
MIN_CONTENT_SIZE = 30         # en dessous, un chunk est éliminé (bruit : numéro de page isolé, etc.)
MIN_PAGE_SIZE_FOR_SPLIT = 300 # en dessous, la page est gardée telle quelle sans passer par le LLM
TOC_DOT_RATIO = 0.3          # proportion de lignes à points de suspension pour détecter une table des matières

# Marqueur que le LLM doit insérer entre les sections logiques
SPLIT_MARKER = "===SPLIT==="

# Prompt pour le split agentic — le LLM insère des marqueurs sans modifier le texte
SPLIT_PROMPT = """Texte d'une page de brochure universitaire :
---
{page_text}
---

Insère le marqueur ===SPLIT=== entre chaque section logique distincte (changement de sujet, nouveau paragraphe thématique, nouveau tableau).
Ne modifie pas le texte. Insère uniquement des marqueurs ===SPLIT=== aux endroits appropriés.
Ne coupe jamais à l'intérieur d'un tableau (lignes commençant par |).
Si la page entière porte sur un seul sujet, ne mets aucun marqueur.

Texte avec marqueurs :"""

# Stopwords français pour l'extraction déterministe de mots-clés
_STOPWORDS = frozenset(
    "le la les de du des un une et en est pour dans par sur avec qui que au aux"
    " ce cette ces son sa ses leur leurs nous vous ils elles on ne pas plus ou"
    " tout tous toute toutes autre autres même être avoir fait faire peut sont"
    " a été sera entre dont aussi bien très comme lors d l qu s n c y".split()
)


def _load_pdf(pdf_path: Path) -> list[Document]:
    """
    Convertit un PDF en Markdown via pymupdf4llm, une page à la fois.
    Chaque page devient un Document indépendant avec son numéro de page en métadonnée.
    Les pages vides (couverture, pages blanches) sont ignorées.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    documents = []
    for page in pages:
        text = ftfy.fix_text(page["text"])  # répare les accents cassés (ex: `a → à, ´e → é)
        if not text.strip():  # ignore les pages sans contenu (couvertures, pages blanches)
            continue
        page_num = page["metadata"].get("page_number", 0)  # numéro de page 0-indexé
        documents.append(Document(
            page_content=text,
            metadata={"source": pdf_path.name, "page": page_num + 1},  # +1 pour afficher en 1-indexé
        ))
    return documents


def load_documents() -> list[Document]:
    """Charge tous les fichiers supportés depuis le dossier documents/."""
    documents = []
    for file_path in sorted(DOCUMENTS_DIR.iterdir()):  # ordre alphabétique pour la reproductibilité
        if file_path.name.startswith("."):  # ignore les fichiers cachés (.DS_Store, etc.)
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded = loader.load()
        elif suffix == ".pdf":
            loaded = _load_pdf(file_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
            loaded = loader.load()
        else:
            print(f"Format ignoré : {file_path.name}")
            continue
        # Force le nom du fichier dans les métadonnées pour savoir d'où vient chaque chunk
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        documents.extend(loaded)
        print(f"  ✓ {file_path.name} ({len(loaded)} doc(s))")
    return documents


def _is_toc_page(text: str) -> bool:
    """
    Détecte une page de table des matières en comptant les lignes contenant des points de
    suspension ('. . .' ou '...'), caractéristiques des entrées de TdM.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    dot_lines = sum(1 for l in lines if ". . ." in l or "..." in l)
    return dot_lines / len(lines) > TOC_DOT_RATIO


def _clean_page_text(text: str) -> str:
    """
    Nettoie le Markdown brut d'une page avant de le passer au LLM :
    - déduplique les lignes consécutives identiques (ex: "BROCHURE ENSEIGNEMENT 2024 2025" répété)
    - supprime les lignes vides en excès (max 2 consécutives)
    """
    lines = text.splitlines()
    cleaned = []
    prev_line = None
    for line in lines:
        stripped = line.strip()
        # Déduplique les lignes consécutives identiques
        if stripped and stripped == prev_line:
            continue
        cleaned.append(line)
        prev_line = stripped
    result = "\n".join(cleaned)
    # Réduit les séquences de plus de 2 lignes vides à 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _merge_cross_page_text(documents: list[Document]) -> list[Document]:
    """
    Détecte quand une phrase est coupée entre deux pages (la page N ne finit pas par
    une ponctuation de fin de phrase) et fusionne le début de la page N+1 dans la page N.
    Le texte fusionné reste aussi dans la page N+1 — le LLM décidera où couper.
    """
    # Ponctuation qui marque une fin de section (on ne fusionne pas dans ce cas)
    end_markers = re.compile(r"[.;:!?)\]»]$")
    # Marqueurs de page insérés par pymupdf4llm (ex: "PAGE 19 SUR 78", "— 12 —")
    page_marker = re.compile(
        r"(PAGE\s+\d+\s+SUR\s+\d+|—\s*\d+\s*—|\*\*-\s*BROCHURE\b.*|\d{1,3}\s*$)",
        re.IGNORECASE,
    )

    for i in range(len(documents) - 1):
        # Même source uniquement (pas de fusion entre deux PDF différents)
        if documents[i].metadata.get("source") != documents[i + 1].metadata.get("source"):
            continue

        # Dernière ligne non vide, en ignorant les marqueurs de page
        lines = [l.strip() for l in documents[i].page_content.rstrip().splitlines() if l.strip()]
        last_content_line = ""
        for line in reversed(lines):
            if not page_marker.match(line):
                last_content_line = line
                break

        # Si la page ne finit pas par une ponctuation de fin → la phrase continue sur la page suivante
        if last_content_line and not end_markers.search(last_content_line):
            next_text = documents[i + 1].page_content
            # Prend le début de la page suivante jusqu'au premier titre ## ou double saut de ligne
            boundary = len(next_text)
            for pattern in ["\n## ", "\n### ", "\n\n"]:
                pos = next_text.find(pattern)
                if pos > 0:
                    boundary = min(boundary, pos)
            continuation = next_text[:boundary].strip()
            if continuation:
                documents[i].page_content += "\n" + continuation

    return documents


def _restart_ollama():
    """
    Tue et relance le serveur Ollama depuis /tmp (contournement pour Colab) :
    llama-server peut crasher en cours de route sans qu'Ollama ne le redémarre.
    """
    os.system("pkill -f ollama 2>/dev/null; pkill -f llama-server 2>/dev/null")
    time.sleep(2)
    subprocess.Popen(
        ["ollama", "serve"],
        cwd="/tmp",
        stdout=open("/tmp/ollama.log", "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(5)


def _invoke_with_retry(llm: OllamaLLM, prompt: str, retries: int = 3) -> str:
    """Réessaie l'appel LLM en redémarrant Ollama entre chaque tentative en cas de crash."""
    for attempt in range(retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if attempt < retries:
                print(f"    ! Erreur LLM ({e}) — redémarrage et nouvelle tentative ({attempt + 1}/{retries})...", flush=True)
                _restart_ollama()
            else:
                raise


def _agentic_split_page(page_text: str, llm: OllamaLLM) -> list[str]:
    """
    Envoie le texte d'une page au LLM qui insère des marqueurs ===SPLIT=== entre les
    sections logiques. Validation : si le LLM a trop modifié le texte (< 60% préservé),
    on retourne le texte original sans découpe.
    """
    # Pages courtes : pas besoin du LLM
    if len(page_text.strip()) < MIN_PAGE_SIZE_FOR_SPLIT:
        return [page_text.strip()]

    prompt = SPLIT_PROMPT.format(page_text=page_text)
    response = _invoke_with_retry(llm, prompt)

    # Validation : le LLM ne doit pas avoir trop modifié le texte
    original_words = set(page_text.lower().split())
    response_words = set(response.lower().replace(SPLIT_MARKER.lower(), "").split())
    if original_words:
        preserved = len(original_words & response_words) / len(original_words)
    else:
        preserved = 0

    if preserved < 0.6:
        # Le LLM a trop modifié le texte — fallback sur le texte original
        return [page_text.strip()]

    # Découpe sur les marqueurs
    segments = response.split(SPLIT_MARKER)
    segments = [s.strip() for s in segments if s.strip()]

    if not segments:
        return [page_text.strip()]

    return segments


def _fallback_split_large(text: str) -> list[str]:
    """
    Découpe déterministe sans overlap pour les chunks encore trop gros après le split LLM.
    Protège les lignes de tableau Markdown (\n|).
    """
    if len(text) <= MAX_CHUNK_SIZE:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=0,
        separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", " "],
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


def _extract_section_path(text: str) -> str:
    """
    Extrait la hiérarchie de titres Markdown présente dans le texte du chunk.
    Ex: "## Master > ### Semestre 1" → "Master > Semestre 1"
    """
    headers = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.+)", line)
        if match:
            title = match.group(2).strip().strip("*")  # retire le gras Markdown **...**
            if title and title not in headers:
                headers.append(title)
    return " > ".join(headers)


def _extract_keywords_deterministic(text: str) -> str:
    """
    Extrait les mots-clés les plus fréquents du chunk par comptage de fréquence,
    après suppression des stopwords français. Zéro hallucination.
    """
    # Retire la ligne de contexte [source | ...] si déjà présente
    clean = text
    if clean.startswith("["):
        end = clean.find("]\n\n")
        if end != -1:
            clean = clean[end + 3:]
    # Retire les marqueurs de page et les symboles Markdown
    clean = re.sub(r"^#{1,3}\s+", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", clean)
    clean = re.sub(r"[|_\-=\[\](){}<>#+*/\\]", " ", clean)

    # Tokenise en mots de 3+ caractères, filtre les stopwords et les nombres purs
    words = re.findall(r"[a-zàâäéèêëïîôùûüÿçœæ]{3,}", clean.lower())
    words = [w for w in words if w not in _STOPWORDS]

    # Bigrams : paires de mots consécutifs (plus informatifs que les unigrams seuls)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    # Compte les fréquences (bigrams + unigrams)
    freq = Counter(bigrams) + Counter(words)
    # Prend les 5 termes les plus fréquents
    top = [term for term, _ in freq.most_common(8) if freq[term] >= 1][:5]

    return ", ".join(top)


def _contextualize_chunk(chunk: Document, section_path: str, keywords: str) -> Document:
    """
    Préfixe le chunk avec une ligne de contexte [source | p.X | section | mots-clés].
    L'embedding et BM25 ne lisent que le texte — cette ligne rend les métadonnées
    visibles pour la recherche.
    """
    source = chunk.metadata.get("source", "document inconnu")
    page = chunk.metadata.get("page", "?")
    parts = [source, f"p.{page}"]
    if section_path:
        parts.append(section_path)
    if keywords:
        parts.append(keywords)
    context_line = "[" + " | ".join(parts) + "]"
    new_content = f"{context_line}\n\n{chunk.page_content}"
    return Document(page_content=new_content, metadata=chunk.metadata)


def _split_documents(documents: list[Document]) -> list[Document]:
    """
    Pipeline de découpage agentic en 7 étapes :
    0. Filtre les pages de table des matières
    1. Nettoie le texte (déduplique le bruit)
    2. Fusionne les phrases coupées entre pages
    3. Split agentic : le LLM décide où couper (===SPLIT===)
    4. Fallback déterministe pour les chunks > 2000 caractères
    5. Filtre les chunks trop courts (< 30 chars)
    6. Assigne chunk_index par page (requis par agent.py pour l'expansion de troncature)
    7. Contextualise : extraction déterministe de section_path + mots-clés, ligne de contexte
    """
    # num_predict élevé : le LLM doit répondre le texte complet de la page avec les marqueurs
    split_llm = OllamaLLM(model=SPLIT_MODEL, num_ctx=4096, temperature=0, num_predict=2048)

    # Reprise sur crash via checkpoint
    checkpoint_path = VECTOR_DB_DIR.parent / "ingest_checkpoint.pkl"
    all_chunks = []
    start_index = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        # Vérifie la version du pipeline pour invalider les anciens checkpoints
        if checkpoint.get("pipeline_version") == 2:
            all_chunks = checkpoint["chunks"]
            start_index = checkpoint["next_index"]
            print(f"  Reprise du checkpoint : {len(all_chunks)} chunk(s) déjà prêts, "
                  f"document {start_index + 1}/{len(documents)}.", flush=True)
        else:
            print("  Ancien checkpoint détecté (pipeline v1) — on repart de zéro.", flush=True)

    for i, doc in enumerate(documents):
        if i < start_index:
            continue

        # Étape 0 : ignore les pages de table des matières
        if _is_toc_page(doc.page_content):
            print(f"  [{i+1}/{len(documents)}] {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} "
                  f"→ table des matières ignorée", flush=True)
        else:
            # Étape 1 : nettoyage du texte
            cleaned_text = _clean_page_text(doc.page_content)

            # Étape 3 : split agentic — le LLM insère des marqueurs entre les sections logiques
            segments = _agentic_split_page(cleaned_text, split_llm)

            # Étape 4 : fallback pour les segments trop gros
            final_segments = []
            for seg in segments:
                final_segments.extend(_fallback_split_large(seg))

            # Étape 5 : filtre les micro-chunks
            final_segments = [s for s in final_segments if len(s.strip()) >= MIN_CONTENT_SIZE]

            # Étape 6 : crée les Documents avec métadonnées + chunk_index
            page_chunks = []
            for idx, seg in enumerate(final_segments):
                chunk = Document(
                    page_content=seg,
                    metadata={**doc.metadata, "chunk_index": idx},
                )
                page_chunks.append(chunk)

            # Étape 7 : contextualisation déterministe (section_path + mots-clés + ligne de contexte)
            for j, chunk in enumerate(page_chunks):
                section_path = _extract_section_path(chunk.page_content)
                keywords = _extract_keywords_deterministic(chunk.page_content)
                page_chunks[j] = _contextualize_chunk(chunk, section_path, keywords)

            all_chunks.extend(page_chunks)

            if page_chunks:
                print(f"  [{i+1}/{len(documents)}] {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} "
                      f"→ {len(page_chunks)} chunk(s)", flush=True)

        # Checkpoint : sauvegarde la progression après chaque page
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "next_index": i + 1, "pipeline_version": 2}, f)

    return all_chunks


def main():
    print("Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("Aucun document trouvé dans le dossier documents/.")
        return
    print(f"\n{len(documents)} document(s) chargé(s).")

    # Étape 2 : fusion inter-pages (avant le split, pour que le LLM voie les phrases complètes)
    documents = _merge_cross_page_text(documents)

    chunks = _split_documents(documents)
    print(f"{len(chunks)} chunk(s) créé(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = 50  # envoi par lots pour ne pas saturer Ollama
    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Lot {i // batch_size + 1} / {-(-len(chunks) // batch_size)} ({len(batch)} chunks)...", flush=True)
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=str(VECTOR_DB_DIR))
        else:
            db.add_documents(batch)
    print("Index créé dans vector_db/.")

    checkpoint_path = VECTOR_DB_DIR.parent / "ingest_checkpoint.pkl"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
