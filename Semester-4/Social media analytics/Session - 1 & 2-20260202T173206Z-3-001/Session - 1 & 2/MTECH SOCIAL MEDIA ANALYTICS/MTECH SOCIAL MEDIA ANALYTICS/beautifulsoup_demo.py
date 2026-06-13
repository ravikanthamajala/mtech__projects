import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

title = soup.find("h1").text
price = soup.find("p", class_="price_color").text
rating = soup.find("p", class_="star-rating")["class"][1]
description = soup.find("meta", {"name":"description"})["content"]

print("Title:", title)
print("Price:", price)
print("Rating:", rating)
print("Description:", description)