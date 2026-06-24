import chromadb
import ollama
import requests
from bs4 import BeautifulSoup
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import fetch_url, extract

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

start_time = time.time()
urls = ["https://www.mpa.gov.sg/who-we-are/about-mpa/mission-vision-values", 
         "https://www.mpa.gov.sg/who-we-are/about-mpa/board-members", 
         "https://www.mpa.gov.sg/maritime-singapore/industry-transformation"]   

all_text = ""
for url in urls: 
    downloaded = fetch_url(url)
    all_text += extract(downloaded, include_images=True, include_links=True, include_formatting=True)

#     response = requests.get(url)

#     soup = BeautifulSoup(response.text, "html.parser")

#     article = soup.find("article")

#     all_text += article.get_text("\n", strip=True)

# Read knowledge file
with open("knowledge.txt", "r", encoding="utf-8") as f:
    all_text += f.read()

chunks = splitter.split_text(all_text)
print(all_text)
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