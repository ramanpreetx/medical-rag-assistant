import requests
from bs4 import BeautifulSoup

BASE_URL = "https://medlineplus.gov"
SITEMAP_URL = "https://medlineplus.gov/sitemap.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_category_urls():

    response = requests.get(
        SITEMAP_URL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    topic_dt = soup.find(
        "dt",
        string=lambda s: s and "Topic by broad groups" in s
    )

    if topic_dt is None:
        raise RuntimeError("Couldn't find 'Topic by broad groups'.")

    topic_dd = topic_dt.find_next_sibling("dd")

    category_urls = []

    seen = set()

    for a in topic_dd.find_all("a", href=True):

        href = a["href"]

        if href.startswith("http"):
            url = href
        else:
            url = BASE_URL + "/" + href.lstrip("/")

        if url in seen:
            continue

        seen.add(url)

        category_urls.append(
            {
                "category": a.get_text(strip=True),
                "url": url,
            }
        )

    return category_urls

if __name__ == "__main__":

    urls = get_category_urls()

    print(f"\nFound {len(urls)} categories\n")

    for url in urls:
        print(url)