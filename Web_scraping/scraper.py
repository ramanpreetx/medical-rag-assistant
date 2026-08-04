import os
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
import json
import re

BASE_URL = "https://medlineplus.gov"

SAVE_FOLDER = "data\scraped_data"

os.makedirs(SAVE_FOLDER, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

session = requests.Session()
session.headers.update(HEADERS)

def format_list(ul, level=0):

    lines = []

    for li in ul.find_all("li", recursive=False):

        indent = "  " * level

        text = []

        for child in li.contents:

            if isinstance(child, str):
                text.append(child.strip())

            elif child.name not in ["ul", "ol"]:
                t = child.get_text(" ", strip=True)
                if t:
                    text.append(t)

        text = " ".join(text).strip()

        if text:
            lines.append(f"{indent}- {text}")


        for nested in li.find_all(["ul", "ol"], recursive=False):
            lines.extend(format_list(nested, level + 1))

    return lines


def html_to_markdown(section):

    lines = []

    for child in section.children:

        if not isinstance(child, Tag):
            continue


        if child.name in ["h2", "h3", "h4"]:

            heading = child.get_text(" ", strip=True)

            if heading.lower() == "summary":
                continue

            if child.name == "h2":
                lines.append(f"\n## {heading}\n")

            elif child.name == "h3":
                lines.append(f"\n### {heading}\n")

            elif child.name == "h4":
                lines.append(f"\n#### {heading}\n")


        elif child.name == "p":

            text = child.get_text(" ", strip=True)

            if text:
                lines.append(text)
                lines.append("")


        elif child.name == "ul":

            lines.extend(format_list(child))
            lines.append("")


        elif child.name == "ol":

            for i, li in enumerate(child.find_all("li", recursive=False), 1):

                text = li.get_text(" ", strip=True)

                lines.append(f"{i}. {text}")

            lines.append("")


        elif child.name == "table":

            rows = []

            for tr in child.find_all("tr"):

                cells = [
                    cell.get_text(" ", strip=True)
                    for cell in tr.find_all(["th", "td"])
                ]

                if cells:
                    rows.append(cells)

            if rows:

                lines.append(
                    "| " + " | ".join(rows[0]) + " |"
                )

                lines.append(
                    "| " + " | ".join(["---"] * len(rows[0])) + " |"
                )

                for row in rows[1:]:

                    lines.append(
                        "| " + " | ".join(row) + " |"
            )

                lines.append("")


        elif child.name == "div":

            lines.extend(html_to_markdown(child))

    return lines

def extract_summary(url):

    response = session.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    summary = soup.find(
        "section",
        id="topsum_section"
    )

    if summary is None:
        raise ValueError("Summary section not found.")

    # Remove unwanted tags
    for tag in summary.find_all(["script", "style"]):
        tag.decompose()

# Remove source attribution
    for tag in summary.find_all("p", class_="attribution"):
        tag.decompose()

    lines = html_to_markdown(summary)

    return "\n".join(lines).strip()

def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name.lower()

def save_page(topic):

    url = topic["url"]
    title = topic["title"]

    print(f"Downloading: {title}")

    text = extract_summary(url)

    filename = sanitize_filename(title) + ".md"

    filepath = os.path.join(
        SAVE_FOLDER,
        filename
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

def main():

    with open(
        "data/topics.json",
        "r",
        encoding="utf-8"
    ) as f:

        topics = json.load(f)

    print(f"Found {len(topics)} topics.\n")

    for topic in tqdm(topics):

        try:
            save_page(topic)

        except Exception as e:

            print(f"Failed: {topic['url']}")
            print(e)

    print("\nDone!")


if __name__ == "__main__":
    main()

    