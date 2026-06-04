from pathlib import Path  # Pour manipuler les chemins de fichiers de façon cross-platform

import pymupdf4llm  # Convertit les PDF en Markdown structuré (meilleur que PyMuPDF brut)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Découpe les textes en morceaux (chunks)
from langchain_chroma import Chroma  # Base de données vectorielle locale (stocke les embeddings)
from langchain_ollama import OllamaEmbeddings  # Génère les vecteurs numériques (embeddings) via Ollama
from langchain_community.document_loaders import TextLoader, Docx2txtLoader  # Loaders pour .txt et .docx
from langchain_core.documents import Document  # Objet standard LangChain : texte + métadonnées

# Chemins clés du projet, calculés dynamiquement depuis l'emplacement de ce fichier
BASE_DIR = Path(__file__).resolve().parent.parent  # Racine du projet (remonte 2 niveaux depuis src/)
DOCUMENTS_DIR = BASE_DIR / "documents"             # Dossier où l'utilisateur dépose ses fichiers
VECTOR_DB_DIR = BASE_DIR / "vector_db"             # Dossier où Chroma sauvegarde la base vectorielle
EMBED_MODEL = "bge-m3"                             # Modèle d'embedding multilingue (à lancer via Ollama)


def _load_pdf(pdf_path: Path) -> list[Document]:
    # pymupdf4llm convertit le PDF en Markdown en respectant la structure (titres, tableaux, etc.)
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    # On enveloppe le texte dans un Document LangChain avec le nom du fichier comme source
    return [Document(page_content=md_text, metadata={"source": pdf_path.name})]


def load_documents() -> list[Document]:
    documents = []
    # On parcourt tous les fichiers du dossier documents/ dans l'ordre alphabétique
    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if file_path.name.startswith("."):  # Ignore les fichiers cachés (.DS_Store, etc.)
            continue
        suffix = file_path.suffix.lower()  # Extension en minuscules pour la comparaison
        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")  # Charge un fichier texte brut
            loaded = loader.load()
        elif suffix == ".pdf":
            loaded = _load_pdf(file_path)  # Utilise notre fonction spéciale pour les PDF
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))  # Extrait le texte d'un fichier Word
            loaded = loader.load()
        else:
            print(f"Format ignoré : {file_path.name}")  # Prévient si le format n'est pas supporté
            continue
        # On force le champ "source" dans les métadonnées pour savoir d'où vient chaque chunk
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        documents.extend(loaded)  # Ajoute les documents chargés à la liste globale
        print(f"  ✓ {file_path.name} ({len(loaded)} doc(s))")
    return documents


def main():
    print("Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("Aucun document trouvé dans le dossier documents/.")
        return
    print(f"\n{len(documents)} document(s) chargé(s).")

    # RecursiveCharacterTextSplitter coupe le texte en respectant d'abord les séparateurs markdown
    # puis les sauts de paragraphe, de ligne, etc. — cela préserve la cohérence des morceaux
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Taille maximale d'un chunk en caractères
        chunk_overlap=200,   # Chevauchement entre chunks pour ne pas perdre le contexte aux coutures
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # Ordre de priorité des coupures
    )
    chunks = splitter.split_documents(documents)  # Découpe tous les documents en chunks
    print(f"{len(chunks)} chunk(s) créé(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  # Initialise le modèle d'embedding
    batch_size = 50  # On envoie les chunks par lots pour éviter de surcharger Ollama
    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]  # Sous-liste de 50 chunks maximum
        print(f"Lot {i // batch_size + 1} / {-(-len(chunks) // batch_size)} ({len(batch)} chunks)...", flush=True)
        if db is None:
            # Premier lot : crée la base Chroma et la sauvegarde sur disque
            db = Chroma.from_documents(batch, embeddings, persist_directory=str(VECTOR_DB_DIR))
        else:
            # Lots suivants : ajoute simplement les nouveaux documents à la base existante
            db.add_documents(batch)
    print("Index créé dans vector_db/.")


if __name__ == "__main__":
    main()
