from pathlib import Path

import pymupdf4llm  # convertit les PDF en Markdown structuré (meilleur que l'extraction brute PyMuPDF)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # découpe le texte en chunks
from langchain_chroma import Chroma  # base de données vectorielle locale
from langchain_ollama import OllamaEmbeddings  # génère les embeddings via Ollama
from langchain_community.document_loaders import TextLoader, Docx2txtLoader  # loaders pour .txt et .docx
from langchain_core.documents import Document  # objet LangChain : texte + métadonnées

# Chemins calculés dynamiquement depuis l'emplacement de ce fichier
BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
DOCUMENTS_DIR = BASE_DIR / "documents"             # dossier où déposer les fichiers à indexer
VECTOR_DB_DIR = BASE_DIR / "vector_db"             # dossier où Chroma sauvegarde la base vectorielle
EMBED_MODEL = "bge-m3"  # modèle d'embedding multilingue — doit être le même que dans ask.py


def _load_pdf(pdf_path: Path) -> list[Document]:
    """
    Convertit un PDF en Markdown via pymupdf4llm.
    Avantage sur PyMuPDF brut : préserve la structure (titres, tableaux)
    et espace correctement les symboles mathématiques.
    """
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    # Enveloppe le texte dans un Document LangChain avec le nom du fichier comme source
    return [Document(page_content=md_text, metadata={"source": pdf_path.name})]


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


def main():
    print("Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("Aucun document trouvé dans le dossier documents/.")
        return
    print(f"\n{len(documents)} document(s) chargé(s).")

    # RecursiveCharacterTextSplitter respecte la structure Markdown :
    # il coupe en priorité sur les titres (## / ###), puis les paragraphes, puis les lignes
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # taille maximale d'un chunk en caractères
        chunk_overlap=200,  # chevauchement pour éviter de couper une idée entre deux chunks
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # ordre de priorité des coupures
    )
    chunks = splitter.split_documents(documents)
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
