import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

from get_categories import get_category_urls

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def get_topic_urls():

    categories = get_category_urls()

    # Key = URL
    topics = {}

    for category in tqdm(categories):

        category_name = category["category"]
        category_url = category["url"]

        try:

            response = requests.get(
                category_url,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # Find every topic on the page
            for li in soup.select("li.item"):

                a = li.find("a", href=True)

                if a is None:
                    continue

                url = a["href"]

                if not url.startswith("http"):
                    url = "https://medlineplus.gov/" + url.lstrip("/")

                if not url.startswith("https://medlineplus.gov/"):
                    continue

                if not url.endswith(".html"):
                    continue

                title = a.get_text(" ", strip=True)

                # First time we've seen this topic
                if url not in topics:

                    topics[url] = {
                        "title": title,
                        "url": url,
                        "categories": [category_name]
                    }

                # Already exists -> just add another category
                else:

                    if category_name not in topics[url]["categories"]:
                        topics[url]["categories"].append(category_name)

        except Exception as e:

            print(f"Failed: {category_url}")
            print(e)

    return list(topics.values())


if __name__ == "__main__":

    topics = get_topic_urls()

    with open("data/topics.json", "w", encoding="utf-8") as f:
        json.dump(
            topics,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nSaved {len(topics)} topics to topics.json")