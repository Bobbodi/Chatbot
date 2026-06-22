import requests
from bs4 import BeautifulSoup

url = "https://www.mpa.gov.sg/home"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

text = soup.get_text(separator="\n")

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

print(text)