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
CONTEXT_MODEL = "gemma2:2b"  # LLM utilisé pour générer une phrase de contexte par chunk (contextual retrieval)
MIN_CHUNK_SIZE = 400  # en dessous de cette taille (en caractères), un chunk est fusionné avec le suivant

# Demande une phrase courte qui situe le chunk (établissement, section, sujet) à partir de la
# page complète d'où il provient. Cette phrase est ensuite préfixée au chunk avant indexation :
# un chunk isolé ("stage d'au moins 4 mois") ne dit pas de lui-même à quel établissement il
# appartient — l'embedding et BM25 ne lisent jamais les métadonnées, seulement le texte indexé.
CONTEXT_PROMPT = """Voici une page extraite d'un document :
{page}

Voici un passage de cette page qui sera indexé séparément pour la recherche :
{chunk}

Écris une phrase complète (15 mots maximum) qui situe ce passage. La phrase doit mentionner l'établissement (Sorbonne Université ou ENS), le niveau (Master 1 ou Master 2) et le sujet précis du passage. Réponds uniquement par cette phrase, sans préambule ni guillemets."""


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


def _contextualize_chunks(chunks: list[Document], page_text: str, llm: OllamaLLM) -> list[Document]:
    """
    Demande au LLM, pour chaque chunk final, une phrase de contexte à partir de la page
    d'origine, et la préfixe au texte avant indexation (technique de "contextual retrieval").
    Contrairement aux métadonnées (chemin de titres, source), cette phrase fait partie du
    texte indexé : l'embedding et BM25 la "voient" directement, ce qui aide à désambiguïser
    un chunk qui, isolé, ne précise pas son établissement ou son sujet.
    """
    result = []
    for chunk in chunks:
        prompt = CONTEXT_PROMPT.format(page=page_text, chunk=chunk.page_content)
        lines = llm.invoke(prompt).strip().splitlines()
        context_line = lines[0].strip() if lines else ""
        new_content = f"{context_line}\n\n{chunk.page_content}" if context_line else chunk.page_content
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
    5. _contextualize_chunks — ajoute en tête de chaque chunk final une phrase de contexte
       générée par le LLM à partir de sa page d'origine (établissement / section / sujet),
       pour que la recherche dispose d'un signal explicite même sur un chunk isolé.
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
    # num_predict bas : on ne veut qu'une phrase courte, pas une réponse longue (gagne du temps de génération)
    context_llm = OllamaLLM(model=CONTEXT_MODEL, num_ctx=4096, temperature=0, num_predict=60)

    all_chunks = []
    for i, doc in enumerate(documents):
        # Étape 0 : ignore les pages de table des matières
        if _is_toc_page(doc.page_content):
            print(f"  [{i+1}/{len(documents)}] {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} "
                  f"→ table des matières ignorée", flush=True)
            continue

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

        if not sub_chunks:
            continue

        # Étape 5 : ajoute une phrase de contexte générée par le LLM en tête de chaque chunk
        sub_chunks = _contextualize_chunks(sub_chunks, doc.page_content, context_llm)

        all_chunks.extend(sub_chunks)
        print(f"  [{i+1}/{len(documents)}] {doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} "
              f"→ {len(sub_chunks)} chunk(s) contextualisé(s)", flush=True)

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


if __name__ == "__main__":
    main()
