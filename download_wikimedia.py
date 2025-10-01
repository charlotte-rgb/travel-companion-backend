import os
import requests
from tqdm import tqdm

API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "TravelCompanionBot/1.0 (contact: your_email@example.com)"
}

def download_landmark_images(query, out_dir="data/landmarks", limit=50):
    """
    Download images from Wikimedia Commons for a given landmark.
    """
    landmark_dir = os.path.join(out_dir, query.replace(" ", "_"))
    os.makedirs(landmark_dir, exist_ok=True)

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 800,
        "gsrnamespace": 6  # restrict search to media files
    }

    print(f"\n🔎 Searching Wikimedia Commons for '{query}'...")
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("❌ Request failed:", e)
        return

    if "query" not in data or "pages" not in data["query"]:
        print("❌ No images found for:", query)
        return

    pages = data["query"]["pages"]

    print(f"⬇️ Downloading up to {limit} images for {query}...")
    for page in tqdm(pages.values()):
        if "imageinfo" in page:
            url = page["imageinfo"][0]["thumburl"]
            filename = os.path.join(landmark_dir, os.path.basename(url))
            try:
                img_data = requests.get(url, headers=HEADERS, timeout=10).content
                with open(filename, "wb") as f:
                    f.write(img_data)
            except Exception as e:
                print("⚠️ Failed to download:", e)

    print(f"✅ Done. Saved images in {landmark_dir}")


if __name__ == "__main__":
    landmarks = [
    "Aachen Cathedral",
    "Palatine Chapel Aachen",
    "Aachen Cathedral Treasury",
    "Aachen Town Hall",
    "Elisenbrunnen Aachen",
    "Centre Charlemagne Aachen",
    "Marktplatz Aachen",
    "Carolus Thermen Aachen",
    "Suermondt Ludwig Museum Aachen",
    "Couven Museum Aachen",
    "Printen Museum Aachen",
    "Ponttor Aachen",
    "Marschiertor Aachen",
    "Granus Tower Aachen",
    "Bismarck Tower Aachen",
    "Belvedere Water Tower Aachen",
    "Lousberg Aachen"
]


    for lm in landmarks:
        download_landmark_images(lm, out_dir="data/landmarks", limit=50)
