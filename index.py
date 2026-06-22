import chromadb
import ollama

# Read knowledge file
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("\n\n")]

db = chromadb.PersistentClient(path="./chroma_db")

try:
    db.delete_collection("knowledge")
except:
    pass

collection = db.create_collection("knowledge")

for i, chunk in enumerate(chunks):

    embedding = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding["embeddings"][0]]
    )

print("Knowledge base indexed successfully.")