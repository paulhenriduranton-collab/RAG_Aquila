from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBED_MODEL = "nomic-embed-text"


def load_documents():
    documents = []
    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.name.startswith("."):
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyMuPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            print(f"Format ignoré : {file_path.name}")
            continue
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        documents.extend(loaded)
    return documents


def main():
    print("Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("Aucun document trouvé dans le dossier documents/.")
        return
    print(f"{len(documents)} document(s) chargé(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunk(s) créé(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    batch_size = 50
    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Lot {i // batch_size + 1} / {len(chunks) // batch_size + 1} ({len(batch)} chunks)...", flush=True)
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=str(VECTOR_DB_DIR))
        else:
            db.add_documents(batch)
    print("Index créé dans vector_db/.")


if __name__ == "__main__":
    main()
