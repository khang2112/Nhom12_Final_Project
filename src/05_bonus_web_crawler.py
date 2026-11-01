# BONUS WEBSITE: https://books.toscrape.com/catalogue/page-1.html
"""Optional real book crawler (>=300 records, >=8 fields).

Requires network access and respects robots.txt and a 1-second delay.
The resulting CSV is separate from Gapminder and contains no fabricated rows.
"""

import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://books.toscrape.com/"
UA = "University-DataAnalystAI-Project/1.0 (educational; polite 1s delay)"


def get(session, url):
    """Fetch one page, then throttle before parsing its HTML."""
    response = session.get(url, timeout=18)
    response.raise_for_status()
    time.sleep(1.0)
    return BeautifulSoup(response.text, "html.parser")


def main():
    """Collect public product cards and require at least 300 genuine rows."""
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    rules = RobotFileParser()
    rules.set_url(urljoin(BASE, "robots.txt"))
    try:
        rules.read()
    except Exception as error:
        raise RuntimeError("Unable to verify robots policy; abort crawling") from error

    rows = []
    for page in range(1, 36):  # Up to 35 pages of actual book listings.
        url = urljoin(BASE, f"catalogue/page-{page}.html")
        if not rules.can_fetch(UA, url):
            raise PermissionError("robots.txt disallows " + url)
        try:
            document = get(session, url)
        except requests.RequestException as error:
            print("Stopped on inaccessible page:", url, error)
            break

        cards = document.select("article.product_pod")
        if not cards:
            break

        for card in cards:
            link = card.select_one("h3 a")
            if not link:
                continue

            detail_url = urljoin(url, link.get("href", ""))
            rating = next(
                (
                    value
                    for value in card.select_one(".star-rating").get("class", [])
                    if value != "star-rating"
                ),
                "Unknown",
            )
            availability = card.select_one(".availability").get_text(" ", strip=True)
            image = card.select_one("img")

            rows.append(
                {
                    "book_id": urlparse(detail_url).path.rsplit("/", 2)[-2],
                    "title": link.get("title") or link.get_text(" ", strip=True),
                    "price_gbp": float(
                        card.select_one(".price_color")
                        .get_text(strip=True)
                        .replace("£", "")
                    ),
                    "star_rating_text": rating,
                    "in_stock": bool("In stock" in availability),
                    "availability_text": availability,
                    "product_url": detail_url,
                    "image_url": urljoin(url, image.get("src", "")) if image else "",
                    "page_number": page,
                    "source_website": "books.toscrape.com",
                }
            )

    df = pd.DataFrame(rows)
    if len(df) < 300:
        raise RuntimeError(
            f"Only {len(df)} real records. Must rerun crawl to reach >=300; no fake filler."
        )
    if df.product_url.duplicated().any():
        raise ValueError("Duplicate book URLs")

    path = ROOT / "data/raw/bonus_books_scraped.csv"
    df.to_csv(path, index=False)
    print(f"Bonus crawl verified: {len(df)} products, {len(df.columns)} attributes, {path}")


if __name__ == "__main__":
    main()
