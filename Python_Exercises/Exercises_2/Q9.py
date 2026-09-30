import os
import requests
from bs4 import BeautifulSoup

output_dir = os.path.dirname(os.path.abspath(__file__))

url = input("Enter a URL: ")
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

output_path = os.path.join(output_dir, "scraped_data.txt")
with open(output_path, "w", encoding="utf-8") as file:
    file.write(soup.get_text())

print("Scraped data saved to", output_path)
