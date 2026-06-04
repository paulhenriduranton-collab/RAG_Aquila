from pathlib import Path

import pymupdf4llm  # convertit les PDF en Markdown structuré
from langchain_text_splitters import RecursiveCharacterTextSplitter  # découpe le texte en chunks
from langchain_chroma import Chroma  # base de données vectorielle
from langchain_ollama import OllamaEmbeddings  # génère les embeddings via Ollama
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_core.documents import Document  # objet texte + métadonnées

BASE_DIR = Path(__file__).resolve().parent.parent  # racine du projet
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBED_MODEL = "bge-m3"  # modèle d'embedding multilingue


def _load_pdf(pdf_path: Path) -> list[Document]:
    # pymupdf4llm préserve la structure (titres, tableaux) contrairement à une extraction brute
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    return [Document(page_content=md_text, metadata={"source": pdf_path.name})]


def load_documents() -> list[Document]:
    documents = []
    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if file_path.name.startswith("."):  # ignore les fichiers cachés
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
            doc.metadata["source"] = file_path.name  # force le nom du fichier dans les métadonnées
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # chevauchement pour ne pas perdre le contexte aux coutures entre chunks
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # coupe en priorité sur les titres Markdown
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunk(s) créé(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
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


if __name__ == "__main__":
    main()
