import os
import pickle
import subprocess
import time
from pathlib import Path

import ftfy          # répare les encodages cassés dans les textes extraits de PDF
import pymupdf4llm  # convertit les PDF en Markdown structuré (meilleur que l'extraction brute PyMuPDF)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # base de données vectorielle locale
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # génère les embeddings et contextualise les chunks via Ollama
from langchain_community.document_loaders import TextLoader, Docx2txtLoader  # loaders pour .txt et .docx
from langchain_core.documents import Document  # objet LangChain : texte + métadonnées

# Chemins calculés dynamiquement depuis l'emplacement de ce fichier
BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
DOCUMENTS_DIR = BASE_DIR / "documents"             # dossier où déposer les fichiers à indexer
VECTOR_DB_DIR = Path("C:/vector_db_aquila")        # hors OneDrive — SQLite corrompu par la synchro cloud
EMBED_MODEL = "bge-m3"  # modèle d'embedding multilingue — doit être le même que dans ask.py
CONTEXT_MODEL = "gemma2:2b"   # LLM pour générer une phrase de contexte par chunk — gemma2:2b suffisant pour l'extraction de mots-clés, 4x plus rapide que gemma4:12b
MIN_CHUNK_SIZE = 500  # en dessous de cette taille (en caractères), un chunk est fusionné avec le suivant
PAGE_OVERLAP_CHARS = 300  # début de la page suivante recopié en fin de chaque page, pour ne pas couper
                           # une liste/section à cheval sur deux pages (cf. problème "coupure de listes")
FINAL_OVERLAP_CHARS = 200  # overlap copié systématiquement entre tous les chunks adjacents d'une même page

# Demande une liste de mots-clés sur le SUJET PRÉCIS du chunk (le LLM ne voit que le chunk,
# pas le document entier — il ne peut donc pas deviner fiablement l'établissement, le niveau
# ou le parcours si ce n'est pas écrit dans le chunk lui-même). Ces informations structurelles
# sont ajoutées séparément, de façon déterministe, à partir des métadonnées (source, page,
# hiérarchie de titres h1/h2/h3 capturée par MarkdownHeaderTextSplitter) — voir _contextualize_chunks.
CONTEXT_PROMPT = """Passage :
{chunk}

SORTIE ATTENDUE : une liste de 3 à 6 mots-clés séparés par des virgules. Rien d'autre.

EXEMPLES VALIDES :
algèbre linéaire, ECTS, semestre 1, prérequis
stage, durée minimale, validation, étranger
enseignant référent, filière, informatique

EXEMPLES INVALIDES (ne jamais produire) :
Ce passage décrit... → INTERDIT
Ce passage concerne... → INTERDIT
Le texte présente... → INTERDIT
Toute phrase avec un verbe conjugué → INTERDIT

Règles :
- Noms et expressions nominales uniquement, jamais de verbe conjugué.
- Pas de guillemets, pas de tirets, pas de numérotation.
- Ne pas mentionner d'établissement ni de niveau (M1/M2) : non disponible dans ce passage.
Réponds par la liste uniquement."""


def _load_pdf(pdf_path: Path) -> list[Document]:
    """
    Convertit un PDF en Markdown via pymupdf4llm, une page à la fois.
    Chaque page devient un Document indépendant avec son numéro de page en métadonnée.
    Avantage : un tableau qui tient sur une page ne sera jamais coupé entre deux chunks.
    Les pages vides (couverture, pages blanches) sont ignorées.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    documents = []
    for page in pages:
        text = ftfy.fix_text(page["text"])  # répare les accents cassés (ex: `a → à, ´e → é)
        if not text.strip():  # ignore les pages sans contenu (couvertures, pages blanches)
            continue
        page_num = page["metadata"].get("page_number", 0)  # numéro de page 0-indexé (int) — la clé est "page_number", pas "page"
        documents.append(Document(
            page_content=text,
            metadata={"source": pdf_path.name, "page": page_num + 1},  # +1 pour afficher en 1-indexé
        ))

    # Chevauchement entre pages : recopie le début de chaque page à la fin de la précédente,
    # pour qu'une liste/section coupée par un saut de page se retrouve complète dans au moins un chunk.
    for i in range(len(documents) - 1):
        next_start = documents[i + 1].page_content[:PAGE_OVERLAP_CHARS]
        documents[i].page_content += "\n\n" + next_start

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


def _merge_small_chunks(chunks: list[Document], min_size: int = MIN_CHUNK_SIZE) -> list[Document]:
    """
    Fusionne avec son voisin tout chunk trop court (ex: une simple ligne de calendrier
    isolée sous son propre titre, du type "## Fin des cours \n Vendredi 17 janvier 2025").
    Un chunk de 50-100 caractères est trop pauvre en mots pour être bien classé par la
    recherche sémantique/BM25 — il se fait systématiquement déclasser par des chunks plus
    longs et plus riches en mots-clés, même issus d'un autre document.
    """
    merged: list[Document] = []
    buffer: Document | None = None
    for chunk in chunks:
        buffer = chunk if buffer is None else Document(
            page_content=buffer.page_content + "\n\n" + chunk.page_content,
            metadata=buffer.metadata,
        )
        if len(buffer.page_content) >= min_size:
            merged.append(buffer)
            buffer = None
    if buffer is not None:  # reliquat final trop court : on le rattache au chunk précédent
        if merged:
            previous = merged.pop()
            buffer = Document(
                page_content=previous.page_content + "\n\n" + buffer.page_content,
                metadata=previous.metadata,
            )
        merged.append(buffer)
    return merged


def _add_overlap_between_chunks(chunks: list[Document], overlap: int = FINAL_OVERLAP_CHARS) -> list[Document]:
    """
    Copie les `overlap` derniers caractères du chunk i au début du chunk i+1, pour tous
    les chunks adjacents d'une même page. RecursiveCharacterTextSplitter ne produit de
    l'overlap que quand il coupe un chunk > chunk_size — deux chunks courts (< 1000 chars)
    n'ont donc aucun recouvrement entre eux. Sans cet overlap, une information qui tombe
    exactement à la frontière entre deux chunks (ex: "Enseignant référent : Marc Lelarge"
    juste après une description de filière) risque d'être isolée et mal retrouvée.
    Le cross-page est déjà géré par PAGE_OVERLAP_CHARS dans _load_pdf — cette fonction
    ne s'applique qu'à l'intérieur d'une même page.
    Les doublons générés sont filtrés au retrieval par DEDUP_THRESHOLD dans ask.py.
    """
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        # Copie la queue du chunk précédent en tête du chunk courant
        tail = chunks[i - 1].page_content[-overlap:]
        result.append(Document(
            page_content=tail + "\n\n" + chunks[i].page_content,
            metadata=chunks[i].metadata,
        ))
    return result


def _restart_ollama():
    """
    Tue et relance le serveur Ollama depuis /tmp (même contournement que sur Colab) :
    llama-server peut crasher en cours de route ("completion error ... EOF") sur les longs
    runs de contextualisation, sans qu'Ollama ne le redémarre tout seul.
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
    """Réessaie l'appel LLM en redémarrant Ollama entre chaque tentative en cas de crash de llama-server."""
    for attempt in range(retries + 1):  # +1 : la dernière itération est la tentative finale sans redémarrage
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if attempt < retries:
                print(f"    ! Erreur LLM ({e}) — redémarrage et nouvelle tentative ({attempt + 1}/{retries})...", flush=True)
                _restart_ollama()
            else:
                raise  # toutes les tentatives épuisées : on laisse remonter l'erreur


def _contextualize_chunks(chunks: list[Document], llm: OllamaLLM) -> list[Document]:
    """
    Préfixe chaque chunk avec une ligne de contexte avant indexation (technique de
    "contextual retrieval") : l'embedding et BM25 ne lisent jamais les métadonnées,
    seulement le texte indexé — un chunk isolé ne dit pas de lui-même son établissement,
    sa page ou sa section.

    La ligne de contexte combine deux sources :
    - Déterministe (sans LLM, donc sans hallucination) : source, page, chemin de titres
      (h1/h2/h3, capturé par MarkdownHeaderTextSplitter) — donne établissement/niveau/parcours
      quand cette info est dans la structure du document.
    - LLM (CONTEXT_PROMPT) : 3-6 mots-clés sur le sujet précis du chunk, une tâche que le
      LLM peut faire à partir du seul texte du chunk (pas besoin du document entier).
    """
    result = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "document inconnu")  # nom du fichier PDF d'origine
        page = chunk.metadata.get("page", "?")
        section_path = " > ".join(
            chunk.metadata[h] for h in ("h1", "h2", "h3") if chunk.metadata.get(h)
        )

        prompt = CONTEXT_PROMPT.format(chunk=chunk.page_content)
        # Récupère la première ligne non vide ; fallback sur "" si le LLM renvoie une réponse vide
        lines = [l for l in _invoke_with_retry(llm, prompt).strip().splitlines() if l.strip()]
        keywords = lines[0].strip() if lines else ""

        parts = [source, f"p.{page}"]
        if section_path:
            parts.append(section_path)
        if keywords:
            parts.append(keywords)
        context_line = "[" + " | ".join(parts) + "]"

        new_content = f"{context_line}\n\n{chunk.page_content}"
        result.append(Document(page_content=new_content, metadata=chunk.metadata))
    return result


MIN_CONTENT_SIZE = 30  # taille minimale du contenu utile d'un chunk (hors phrase de contexte LLM)
TOC_DOT_RATIO = 0.3   # proportion minimale de lignes avec points de suspension pour détecter une TdM


def _is_toc_page(text: str) -> bool:
    """
    Détecte une page de table des matières en comptant les lignes contenant des points de
    suspension ('. . .' ou '...'), caractéristiques des entrées de TdM type
    '1.1  Objectifs . . . . . . . . . . 7'. Ces pages n'apportent aucune information
    récupérable par la recherche : elles listent uniquement des numéros de pages.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    dot_lines = sum(1 for l in lines if ". . ." in l or "..." in l)
    return dot_lines / len(lines) > TOC_DOT_RATIO


def _split_documents(documents: list[Document]) -> list[Document]:
    """
    Pipeline de découpage en cinq étapes :
    0. Filtre les pages de table des matières (points de suspension > 30 % des lignes) —
       ces pages ne contiennent que des numéros de pages et polluent la recherche.
    1. MarkdownHeaderTextSplitter — coupe sur les titres (##, ###) et met le chemin de titres
       en métadonnée de chaque chunk. Cela permet à l'embedding de savoir dans quelle section
       se trouve le chunk, évitant les confusions entre sections sémantiquement proches.
    2. _merge_small_chunks — recolle les sections trop courtes (ex: lignes de calendrier sous
       des petits titres) avec la suivante, pour éviter des micro-chunks trop pauvres en contexte.
    3. RecursiveCharacterTextSplitter — coupe les sections encore trop longues en sous-chunks
       de 1000 caractères, en protégeant les lignes de tableaux Markdown (\n|).
    4. Filtre les chunks dont le contenu utile est trop court (< MIN_CONTENT_SIZE) — évite
       d'envoyer au LLM des chunks ne contenant qu'un numéro de page ou un symbole isolé.
    4b. _add_overlap_between_chunks — copie les FINAL_OVERLAP_CHARS derniers chars du chunk i
       au début du chunk i+1 pour tous les chunks adjacents, intra ET inter-page. Appliqué
       après la boucle sur tous les all_chunks finaux. Évite de perdre une information à la
       frontière entre deux chunks (ex: "Enseignant référent : Marc Lelarge" après une
       description de filière).
    5. _contextualize_chunks — ajoute en tête de chaque chunk final une ligne de contexte
       combinant métadonnées déterministes (source, page, chemin de titres) et mots-clés
       générés par le LLM sur le sujet précis du chunk, pour que la recherche dispose d'un
       signal explicite même sur un chunk isolé.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,  # garde les titres dans le texte pour que l'embedding les voie
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n|", "\n", " ", ""],
    )
    # num_predict bas : on ne veut qu'une courte liste de mots-clés, pas une réponse longue (gagne du temps de génération)
    context_llm = OllamaLLM(model=CONTEXT_MODEL, num_ctx=4096, temperature=0, num_predict=40)

    # Reprise sur crash : si un run précédent a été interrompu (ex: crash llama-server),
    # on repart des chunks déjà contextualisés au lieu de tout refaire depuis le début.
    checkpoint_path = VECTOR_DB_DIR.parent / "ingest_checkpoint.pkl"
    all_chunks = []
    start_index = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        all_chunks = checkpoint["chunks"]
        start_index = checkpoint["next_index"]
        print(f"  Reprise du checkpoint : {len(all_chunks)} chunk(s) déjà prêts, "
              f"document {start_index + 1}/{len(documents)}.", flush=True)

    for i, doc in enumerate(documents):
        if i < start_index:
            continue

        # Étape 0 : ignore les pages de table des matières
        if _is_toc_page(doc.page_content):
            print(f"  [{i+1}/{len(documents)}] {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} "
                  f"→ table des matières ignorée", flush=True)
        else:
            # Étape 1 : découpe par titres → chaque section garde le contexte de son titre
            header_chunks = header_splitter.split_text(doc.page_content)
            for hc in header_chunks:
                hc.metadata.update(doc.metadata)  # recopie source + page dans chaque section

            # Étape 2 : fusionne les sections trop courtes avec leur voisine
            merged_chunks = _merge_small_chunks(header_chunks)

            # Étape 3 : découpe par taille si une section dépasse 1000 caractères
            sub_chunks = size_splitter.split_documents(merged_chunks)

            # Étape 4 : filtre les chunks dont le contenu utile est trop court
            sub_chunks = [c for c in sub_chunks if len(c.page_content.strip()) >= MIN_CONTENT_SIZE]

            if sub_chunks:
                # Étape 5 : ajoute une ligne de contexte (métadonnées + mots-clés LLM) en tête de chaque chunk
                sub_chunks = _contextualize_chunks(sub_chunks, context_llm)
                all_chunks.extend(sub_chunks)
                print(f"  [{i+1}/{len(documents)}] {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} "
                      f"→ {len(sub_chunks)} chunk(s) contextualisé(s)", flush=True)

        # Checkpoint : sauvegarde la progression pour pouvoir reprendre après un crash llama-server
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "next_index": i + 1}, f)

    # Étape 4b : overlap systématique entre tous les chunks adjacents (intra ET inter-page).
    # Appliqué après la boucle pour couvrir aussi les frontières entre pages. La queue copiée
    # est les FINAL_OVERLAP_CHARS derniers chars de contenu du chunk i (le préfixe de contexte
    # "[source | p.X | ...]" est en tête de chunk, donc la queue est du contenu pur).
    all_chunks = _add_overlap_between_chunks(all_chunks)

    return all_chunks


def main():
    print("Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("Aucun document trouvé dans le dossier documents/.")
        return
    print(f"\n{len(documents)} document(s) chargé(s).")

    chunks = _split_documents(documents)
    print(f"{len(chunks)} chunk(s) créé(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)  # les bindings Rust de ChromaDB ne créent pas le dossier s'il n'existe pas
    batch_size = 50  # envoi par lots pour ne pas saturer Ollama (évite les timeouts sur grosses bases)
    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Lot {i // batch_size + 1} / {-(-len(chunks) // batch_size)} ({len(batch)} chunks)...", flush=True)
        if db is None:
            # Premier lot : crée la base Chroma et la sauvegarde sur disque
            db = Chroma.from_documents(batch, embeddings, persist_directory=str(VECTOR_DB_DIR))
        else:
            # Lots suivants : ajoute les nouveaux documents à la base existante
            db.add_documents(batch)
    print("Index créé dans vector_db/.")

    checkpoint_path = VECTOR_DB_DIR.parent / "ingest_checkpoint.pkl"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
