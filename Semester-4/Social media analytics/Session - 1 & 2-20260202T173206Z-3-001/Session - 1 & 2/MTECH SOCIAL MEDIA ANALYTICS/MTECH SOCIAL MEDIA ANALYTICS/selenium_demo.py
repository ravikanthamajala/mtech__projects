from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Basic Selenium Setup
options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# Scrape a real website
driver.get("https://quotes.toscrape.com/")

quotes = driver.find_elements("class name", "text")

print("\nScraped Quotes:\n")
for q in quotes[:5]:
    print("-", q.text)

driver.quit()