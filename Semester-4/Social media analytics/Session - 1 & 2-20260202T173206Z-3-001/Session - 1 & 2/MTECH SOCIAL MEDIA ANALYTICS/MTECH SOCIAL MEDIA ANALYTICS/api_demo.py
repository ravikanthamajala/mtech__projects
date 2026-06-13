import requests

url = "https://dummyjson.com/products/1"
response = requests.get(url)

data = response.json()

print("Product:", data["title"])
print("Description:", data["description"])
print("\nReviews:")

for review in data["reviews"]:
    print("-", review["comment"])