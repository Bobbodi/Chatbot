import chromadb
import ollama
import requests
from bs4 import BeautifulSoup
import time

def chunk_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

start_time = time.time()
url = "https://www.mpa.gov.sg/home"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

text = soup.get_text(separator="\n")

# Read knowledge file
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text += f.read()

chunks = chunk_text(text)

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
end_time = time.time() - start_time
print("Knowledge base indexed successfully.")
print("Time taken: {:.2f} s".format(end_time))