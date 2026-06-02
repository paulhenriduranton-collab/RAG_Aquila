from langchain_ollama import OllamaEmbeddings
emb = OllamaEmbeddings(model="nomic-embed-text")
result = emb.embed_query("test")
print("OK, vecteur de taille", len(result))
