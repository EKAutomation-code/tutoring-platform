import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
BASE_URL = "https://scrapeme.live/shop/page/{}/"
headers = {
    "User-Agent": "Mozilla/5.0"
}
data = []
page = 1
while True:
    url = BASE_URL.format(page)
    print(f"Scraping page {page}...")
    try:
        res = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Request error on page {page}: {e}")
        break
    if res.status_code != 200:
        print(f"Stopped: status code {res.status_code} on page {page}")
        break
    soup = BeautifulSoup(res.text, "html.parser")
    products = soup.find_all("li", class_="product")
    if not products:
        print("No products found, stopping scraper.")
        break
    for p in products:
        title_tag = p.find("h2", class_="woocommerce-loop-product__title")
        price_tag = p.find("span", class_="woocommerce-Price-amount")
        link_tag = p.find("a")
        title = title_tag.text.strip() if title_tag else "N/A"
        price = price_tag.text.strip() if price_tag else "N/A"
        link = link_tag["href"] if link_tag else "N/A"
        data.append({
            "title": title,
            "price": price,
            "link": link
        })
    print(f"Page {page} scraped, items: {len(products)}")
    page += 1
    time.sleep(1)
df = pd.DataFrame(data)
df.to_csv("shop_products.csv", index=False)
print(f"DONE: scraped {len(data)} products -> shop_products.csv")
