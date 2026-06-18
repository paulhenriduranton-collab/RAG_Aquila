# Exporte tous les chunks de la base vectorielle ChromaDB dans un fichier Markdown lisible.
# Usage : python export_chunks.py
# Résultat : data/all_chunks.md

import chromadb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "vector_db"
OUTPUT = Path(__file__).resolve().parent / "data" / "all_chunks.md"

# Connexion à la base ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))
col = client.list_collections()[0]
result = col.get(include=["metadatas", "documents"])

# Tri par source → page → chunk_index pour une lecture dans l'ordre du document
entries = list(zip(result["ids"], result["metadatas"], result["documents"]))
entries.sort(key=lambda e: (
    e[1].get("source", ""),
    e[1].get("page", 0),
    e[1].get("chunk_index", 0),
))

lines = [f"# Tous les chunks ({len(entries)})\n"]

current_source = None
for i, (id_, meta, text) in enumerate(entries, 1):
    source = meta.get("source", "?")
    page = meta.get("page", "?")
    idx = meta.get("chunk_index", "?")
    h1 = meta.get("h1", "")
    h2 = meta.get("h2", "")
    h3 = meta.get("h3", "")
    headers = " > ".join(h for h in [h1, h2, h3] if h)

    # Séparateur visuel quand on change de document source
    if source != current_source:
        lines.append(f"\n---\n")
        lines.append(f"# 📄 {source}\n")
        current_source = source

    lines.append(f"---\n")
    lines.append(f"### Chunk {i} / {len(entries)}\n")
    lines.append(f"| Propriété | Valeur |")
    lines.append(f"|-----------|--------|")
    lines.append(f"| Source | {source} |")
    lines.append(f"| Page | {page} |")
    lines.append(f"| Index dans la page | {idx} |")
    if headers:
        lines.append(f"| Titres | {headers} |")
    lines.append(f"| Taille | {len(text)} caractères |")
    lines.append(f"| ID ChromaDB | `{id_}` |")
    lines.append(f"\n**Contenu :**\n")
    lines.append(f"```text\n{text}\n```\n")

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"{len(entries)} chunks exportes dans {OUTPUT}")
