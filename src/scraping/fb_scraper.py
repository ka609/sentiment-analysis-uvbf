from facebook_scraper import get_posts
import pandas as pd
import os

def scrape_facebook(page="uvburkina", pages=5, output_file="data/raw/facebook_posts.csv"):
    """
    Scrape les posts publics d'une page Facebook et sauvegarde en CSV.
    """
    print(f"[INFO] Scraping Facebook page: {page} ({pages} pages)")
    data = []

    for post in get_posts(
        page,
        pages=pages,
        extra_info=True,
        facebook_language="en-US",
        cookies="src/scraping/cookies.txt"
    ):
        data.append({
            "text": post.get("text", ""),
            "time": post.get("time"),
            "likes": post.get("likes", 0),
            "comments": post.get("comments", 0),
            "shares": post.get("shares", 0),
            "post_id": post.get("post_id", "")
        })

    # Créer dossier si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Sauvegarde en CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"[INFO] {len(df)} posts sauvegardés dans {output_file}")


if __name__ == "__main__":
    scrape_facebook(page="uvburkina", pages=10)
