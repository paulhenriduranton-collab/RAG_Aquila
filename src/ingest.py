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
SPLIT_MODEL = "gemma4:12b"  # LLM pour le split agentic

# Seuils du pipeline
MAX_CHUNK_SIZE = 1500         # au-delà, un chunk est redécoupé par le fallback déterministe
MIN_CONTENT_SIZE = 30         # en dessous, un chunk est éliminé (bruit)
MIN_SECTION_SIZE_FOR_SPLIT = 800  # en dessous, la section est fusionnée avec la suivante
TOC_DOT_RATIO = 0.3          # proportion de lignes à points de suspension pour détecter une TdM

# Marqueur interne pour retrouver le numéro de page dans le texte concaténé
_PAGE_MARKER_RE = re.compile(r"<!-- PAGE (\d+) -->")

# Marqueur que le LLM doit insérer entre les sections logiques
SPLIT_MARKER = "===SPLIT==="

# Prompt pour le split agentic — travaille sur des sections thématiques (pas des pages)
SPLIT_PROMPT = """Texte d'une section de brochure universitaire :
---
{section_text}
---

Insère le marqueur ===SPLIT=== entre chaque sous-section logique distincte (changement de sujet, nouveau paragraphe thématique).
Ne modifie pas le texte. Insère uniquement des marqueurs ===SPLIT=== aux endroits appropriés.

RÈGLE ABSOLUE : ne coupe JAMAIS à l'intérieur d'un tableau. Un tableau commence par une ligne | et finit quand les lignes | s'arrêtent. Le tableau entier doit rester dans le même bloc.

RÈGLE ABSOLUE : ne coupe JAMAIS au milieu d'une liste (à puces ou numérotée). Si une liste commence (lignes débutant par -, *, •, ou un numéro), elle doit rester entièrement dans le même bloc jusqu'à la fin de la liste.

Si la section entière porte sur un seul sujet, ne mets aucun marqueur.

Texte avec marqueurs :"""

# Stopwords français pour l'extraction déterministe de mots-clés
_STOPWORDS = frozenset(
    "le la les de du des un une et en est pour dans par sur avec qui que au aux"
    " ce cette ces son sa ses leur leurs nous vous ils elles on ne pas plus ou"
    " tout tous toute toutes autre autres même être avoir fait faire peut sont"
    " a été sera entre dont aussi bien très comme lors d l qu s n c y".split()
)


# ---------------------------------------------------------------------------
#  Chargement des documents
# ---------------------------------------------------------------------------

def _load_pdf(pdf_path: Path) -> list[Document]:
    """
    Convertit un PDF en Markdown via pymupdf4llm, une page à la fois.
    Chaque page devient un Document indépendant avec son numéro de page en métadonnée.
    Les pages vides (couverture, pages blanches) sont ignorées.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    documents = []
    for page in pages:
        text = ftfy.fix_text(page["text"])  # répare les accents cassés
        if not text.strip():
            continue
        page_num = page["metadata"].get("page_number", 0)
        documents.append(Document(
            page_content=text,
            metadata={"source": pdf_path.name, "page": page_num + 1},
        ))
    return documents


def load_documents() -> list[Document]:
    """Charge tous les fichiers supportés depuis le dossier documents/."""
    documents = []
    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if file_path.name.startswith("."):
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
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        documents.extend(loaded)
        print(f"  ✓ {file_path.name} ({len(loaded)} doc(s))")
    return documents


# ---------------------------------------------------------------------------
#  Nettoyage et pré-traitement
# ---------------------------------------------------------------------------

def _is_toc_page(text: str) -> bool:
    """Détecte une page de table des matières (>30% de lignes avec points de suspension)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    dot_lines = sum(1 for l in lines if ". . ." in l or "..." in l)
    return dot_lines / len(lines) > TOC_DOT_RATIO


def _clean_text(text: str) -> str:
    """
    Nettoie le Markdown brut :
    - déduplique les lignes consécutives identiques
    - réduit les lignes vides en excès
    """
    lines = text.splitlines()
    cleaned = []
    prev_line = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev_line:
            continue
        cleaned.append(line)
        prev_line = stripped
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _concat_pages(pages: list[Document]) -> tuple[str, str]:
    """
    Concatène toutes les pages d'un même document en un seul texte continu.
    Insère des marqueurs <!-- PAGE X --> entre les pages pour pouvoir retrouver
    le numéro de page de chaque chunk après découpe.
    Retourne (texte_concaténé, source).
    Les pages de table des matières sont exclues.
    """
    source = pages[0].metadata["source"]
    parts = []
    for doc in pages:
        if _is_toc_page(doc.page_content):
            continue
        page_num = doc.metadata["page"]
        cleaned = _clean_text(doc.page_content)
        if cleaned:
            parts.append(f"<!-- PAGE {page_num} -->\n{cleaned}")
    return "\n\n".join(parts), source


def _get_page_for_position(full_text: str, pos: int) -> int:
    """
    Retrouve le numéro de page correspondant à une position dans le texte concaténé
    en cherchant le dernier marqueur <!-- PAGE X --> avant cette position.
    """
    last_page = 1
    for m in _PAGE_MARKER_RE.finditer(full_text):
        if m.start() <= pos:
            last_page = int(m.group(1))
        else:
            break
    return last_page


def _strip_page_markers(text: str) -> str:
    """Retire les marqueurs <!-- PAGE X --> du texte final d'un chunk."""
    return _PAGE_MARKER_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
#  Pré-découpe par titres (déterministe)
# ---------------------------------------------------------------------------

def _is_pseudo_header(header_content: str) -> bool:
    """
    Détecte les faux titres Markdown issus des bordures de tableau PDF.
    pymupdf4llm convertit les cellules de tableau en lignes commençant par ## |,
    ce qui crée des points de coupure artificiels dans _presplit_by_headers.
    """
    # Retire le formatage Markdown (**, _, espaces, |) et vérifie s'il reste du vrai texte
    stripped = re.sub(r"[*_\s|]", "", header_content)
    return len(stripped) < 3


def _presplit_by_headers(full_text: str, source: str) -> list[Document]:
    """
    Découpe le texte concaténé sur les titres Markdown # et ## pour créer des blocs
    thématiques. Chaque bloc peut couvrir plusieurs pages.
    Les métadonnées page et source sont déterminées par le premier marqueur de page du bloc.
    Ignore les pseudo-titres issus des bordures de tableau PDF (## |, ## **|**, etc.).
    """
    # Trouve les positions de tous les titres # et ##, en filtrant les bordures de tableau
    header_pattern = re.compile(r"^(#{1,2})\s+(.+)", re.MULTILINE)
    split_positions = []
    for m in header_pattern.finditer(full_text):
        if _is_pseudo_header(m.group(2)):
            continue
        split_positions.append(m.start())

    # Si aucun titre trouvé, le document entier est un seul bloc
    if not split_positions:
        page = _get_page_for_position(full_text, 0)
        return [Document(
            page_content=_strip_page_markers(full_text),
            metadata={"source": source, "page": page},
        )]

    # Ajoute le début du texte si le premier titre n'est pas au début
    if split_positions[0] > 0:
        split_positions.insert(0, 0)

    sections = []
    for idx, start in enumerate(split_positions):
        end = split_positions[idx + 1] if idx + 1 < len(split_positions) else len(full_text)
        section_text = full_text[start:end]
        clean = _strip_page_markers(section_text)
        if not clean.strip():
            continue
        page = _get_page_for_position(full_text, start)
        sections.append(Document(
            page_content=clean,
            metadata={"source": source, "page": page},
        ))

    # Fusionne les sections trop courtes avec la PRÉCÉDENTE (pas la suivante)
    # pour garder les sous-sections (Program, Bibliography, Horaires) avec leur titre parent
    merged: list[Document] = []
    for section in sections:
        if merged and len(section.page_content) < MIN_SECTION_SIZE_FOR_SPLIT:
            prev = merged[-1]
            merged[-1] = Document(
                page_content=prev.page_content + "\n\n" + section.page_content,
                metadata=prev.metadata,
            )
        else:
            merged.append(section)

    return merged


# ---------------------------------------------------------------------------
#  Split agentic (LLM)
# ---------------------------------------------------------------------------

def _restart_ollama():
    """Tue et relance le serveur Ollama depuis /tmp (contournement pour Colab)."""
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


def _agentic_split_section(section_text: str, llm: OllamaLLM) -> list[str]:
    """
    Envoie le texte d'une section thématique au LLM qui insère des marqueurs ===SPLIT===
    entre les sous-sections logiques.
    Validation : si le LLM a trop modifié le texte (< 60% préservé), fallback sur le texte original.
    """
    if len(section_text.strip()) < MIN_SECTION_SIZE_FOR_SPLIT:
        return [section_text.strip()]

    prompt = SPLIT_PROMPT.format(section_text=section_text)
    response = _invoke_with_retry(llm, prompt)

    # Validation : le LLM ne doit pas avoir trop modifié le texte
    original_words = set(section_text.lower().split())
    response_words = set(response.lower().replace(SPLIT_MARKER.lower(), "").split())
    if original_words:
        preserved = len(original_words & response_words) / len(original_words)
    else:
        preserved = 0

    if preserved < 0.6:
        return [section_text.strip()]

    segments = response.split(SPLIT_MARKER)
    segments = [s.strip() for s in segments if s.strip()]
    return segments if segments else [section_text.strip()]


def _fallback_split_large(text: str) -> list[str]:
    """
    Découpe déterministe sans overlap pour les chunks encore trop gros.
    Protège les lignes de tableau Markdown.
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


# ---------------------------------------------------------------------------
#  Contextualisation déterministe
# ---------------------------------------------------------------------------

def _extract_section_path(text: str) -> str:
    """Extrait la hiérarchie de titres Markdown présente dans le chunk."""
    headers = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.+)", line)
        if match:
            title = match.group(2).strip().strip("*")
            if title and title not in headers:
                headers.append(title)
    return " > ".join(headers)


def _extract_keywords_deterministic(text: str) -> str:
    """
    Extrait les mots-clés les plus fréquents du chunk par comptage de fréquence,
    après suppression des stopwords français. Zéro hallucination.
    """
    clean = text
    if clean.startswith("["):
        end = clean.find("]\n\n")
        if end != -1:
            clean = clean[end + 3:]
    clean = re.sub(r"^#{1,3}\s+", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", clean)
    clean = re.sub(r"[|_\-=\[\](){}<>#+*/\\]", " ", clean)

    words = re.findall(r"[a-zàâäéèêëïîôùûüÿçœæ]{3,}", clean.lower())
    words = [w for w in words if w not in _STOPWORDS]

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    freq = Counter(bigrams) + Counter(words)
    top = [term for term, _ in freq.most_common(8) if freq[term] >= 1][:5]
    return ", ".join(top)


def _contextualize_chunk(chunk: Document, section_path: str, keywords: str) -> Document:
    """Préfixe le chunk avec une ligne de contexte [source | p.X | section | mots-clés]."""
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


# ---------------------------------------------------------------------------
#  Pipeline principal
# ---------------------------------------------------------------------------

def _split_documents(documents: list[Document]) -> list[Document]:
    """
    Pipeline de découpage agentic :
    1. Concatène toutes les pages par document (texte continu, plus de frontières de page)
    2. Pré-découpe par titres # et ## → blocs thématiques pouvant couvrir plusieurs pages
    3. Le LLM affine chaque bloc en sous-sections cohérentes (===SPLIT===)
    4. Fallback déterministe pour les chunks > MAX_CHUNK_SIZE
    5. Filtre les chunks trop courts
    6. Assigne chunk_index par page (requis par agent.py)
    7. Contextualise avec section_path + mots-clés déterministes
    """
    split_llm = OllamaLLM(model=SPLIT_MODEL, num_ctx=4096, temperature=0, num_predict=2048)

    # Regroupe les pages par source (un PDF = un groupe)
    pages_by_source: dict[str, list[Document]] = {}
    for doc in documents:
        src = doc.metadata["source"]
        pages_by_source.setdefault(src, []).append(doc)

    # Reprise sur crash via checkpoint
    checkpoint_path = VECTOR_DB_DIR.parent / "ingest_checkpoint.pkl"
    all_chunks: list[Document] = []
    done_sources: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        if checkpoint.get("pipeline_version") == 3:
            all_chunks = checkpoint["chunks"]
            done_sources = checkpoint["done_sources"]
            print(f"  Reprise du checkpoint : {len(all_chunks)} chunk(s) déjà prêts, "
                  f"{len(done_sources)} source(s) terminée(s).", flush=True)
        else:
            print("  Ancien checkpoint détecté — on repart de zéro.", flush=True)

    for source, pages in pages_by_source.items():
        if source in done_sources:
            continue

        print(f"\n  Traitement de {source} ({len(pages)} pages)...", flush=True)

        # Étape 1 : concatène toutes les pages (filtre les TdM, nettoie)
        full_text, src = _concat_pages(pages)

        # Étape 2 : pré-découpe par titres # et ## → blocs thématiques
        sections = _presplit_by_headers(full_text, src)
        print(f"    {len(sections)} section(s) thématique(s) détectée(s)", flush=True)

        source_chunks = []
        for s_idx, section in enumerate(sections):
            # Étape 3 : le LLM affine chaque section en sous-chunks
            segments = _agentic_split_section(section.page_content, split_llm)

            # Étape 4 : fallback pour les segments trop gros
            final_segments = []
            for seg in segments:
                final_segments.extend(_fallback_split_large(seg))

            # Étape 5 : filtre les micro-chunks
            final_segments = [s for s in final_segments if len(s.strip()) >= MIN_CONTENT_SIZE]

            for seg in final_segments:
                chunk = Document(
                    page_content=seg,
                    metadata={"source": source, "page": section.metadata["page"]},
                )
                source_chunks.append(chunk)

            print(f"    [{s_idx+1}/{len(sections)}] → {len(final_segments)} chunk(s)", flush=True)

        # Étape 6 : assigne chunk_index par page (agent.py utilise chunk_index=0 pour l'expansion)
        chunks_by_page: dict[int, int] = {}
        for chunk in source_chunks:
            page = chunk.metadata["page"]
            idx = chunks_by_page.get(page, 0)
            chunk.metadata["chunk_index"] = idx
            chunks_by_page[page] = idx + 1

        # Étape 7 : contextualisation déterministe
        for j, chunk in enumerate(source_chunks):
            section_path = _extract_section_path(chunk.page_content)
            keywords = _extract_keywords_deterministic(chunk.page_content)
            source_chunks[j] = _contextualize_chunk(chunk, section_path, keywords)

        all_chunks.extend(source_chunks)
        done_sources.add(source)

        # Checkpoint après chaque source complète
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "done_sources": done_sources, "pipeline_version": 3}, f)

        print(f"  {source} terminé : {len(source_chunks)} chunk(s) au total.", flush=True)

    return all_chunks


def main():
    print("Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("Aucun document trouvé dans le dossier documents/.")
        return
    print(f"\n{len(documents)} document(s) chargé(s).")

    chunks = _split_documents(documents)
    print(f"\n{len(chunks)} chunk(s) créé(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = 50
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
