import chromadb
import ollama
import requests
from bs4 import BeautifulSoup
import time
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from trafilatura import fetch_url, extract
from paddleocr import PPStructureV3
from pathlib import Path

start_time = time.time()
# urls = ["https://www.mpa.gov.sg/who-we-are/about-mpa/mission-vision-values", 
#          "https://www.mpa.gov.sg/who-we-are/about-mpa/board-members", 
#          "https://www.mpa.gov.sg/maritime-singapore/industry-transformation",
#          "https://www.mpa.gov.sg/finance-e-services/tariff-fees-and-charges/ocean-going-vessels/port-dues-tariff"]   

all_text = ""
# for url in urls: 
#     downloaded = fetch_url(url)
#     all_text += extract(downloaded, include_images=True, include_links=True, include_formatting=True)

#     response = requests.get(url)

#     soup = BeautifulSoup(response.text, "html.parser")

#     article = soup.find("article")

#     all_text += article.get_text("\n", strip=True)


pdfs = []
for pdf in pdfs: 
    input_file = pdf
    output_path = Path("./page2_output")
    pipeline = PPStructureV3()
    output = pipeline.predict(input=input_file, text_recognition_model_name="en_PP-OCRv4_mobile_rec")
    markdown_list = []
    markdown_images = []
    for res in output: 
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))
    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
    mkd_file_path = output_path / f"{Path(input_file).stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts["markdown_texts"])
        all_text += markdown_texts["markdown_texts"]
    for item in markdown_images: 
        if item: 
            for path, image in item.items(): 
                file_path = output_path / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)

# Read knowledge file
with open("OSCP.md", "r", encoding="utf-8") as f:
    all_text += f.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)
md_header_splits = markdown_splitter.split_text(all_text)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(md_header_splits)

db = chromadb.PersistentClient(path="./chroma_db")

try:
    db.delete_collection("knowledge")
except:
    pass

collection = db.create_collection("knowledge")

for i, chunk in enumerate(chunks):

    embedding = ollama.embed(
        model="nomic-embed-text",
        input=chunk.page_content
    )

    collection.add(
        ids=[str(i)],
        documents=[chunk.page_content],
        embeddings=[embedding["embeddings"][0]],
        metadatas=[chunk.metadata]
    )
end_time = time.time() - start_time
print("Knowledge base indexed successfully.")
print("Time taken: {:.2f} s".format(end_time))