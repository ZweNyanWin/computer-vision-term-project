"""Download Thai carved wooden handicraft images for the classifier dataset.

Images land in data/<class_name>/. After downloading, MANUALLY review every
folder and delete junk - drawings, plastic reproductions, wrong animals,
duplicates, product collages. Clean data matters far more than more data; on the
previous dataset this step was what decided the result.

TODO: the class list below is provisional. Confirm the final categories with the
teacher before investing an afternoon in cleaning them.

Run:
  conda activate cv
  pip install ddgs requests
  python scraper.py
"""

import os
import time

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

IMAGES_PER_CLASS = 120

# class name -> search queries (Thai + English gives better variety)
CLASSES = {
    "frog":     ["thai wooden frog croaking instrument", "wooden guiro frog thailand",
                 "กบไม้ เครื่องดนตรี", "กบไม้แกะสลัก"],
    "elephant": ["thai carved wooden elephant handicraft", "ช้างไม้แกะสลัก"],
    "owl":      ["thai carved wooden owl handicraft", "นกฮูกไม้แกะสลัก"],
    "turtle":   ["thai carved wooden turtle handicraft", "เต่าไม้แกะสลัก"],
    "gecko":    ["thai carved wooden gecko handicraft", "ตุ๊กแกไม้แกะสลัก"],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def download_class(name, queries, limit):
    from ddgs import DDGS

    out_dir = os.path.join(DATA_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    count = 0
    with DDGS() as ddgs:
        for query in queries:
            if count >= limit:
                break
            for result in ddgs.images(query, max_results=limit):
                if count >= limit:
                    break
                url = result.get("image")
                if not url or url in seen:
                    continue
                seen.add(url)
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=10)
                    resp.raise_for_status()
                except Exception:
                    continue
                ext = ".png" if ".png" in url.lower() else ".jpg"
                path = os.path.join(out_dir, f"{name}_{count:04d}{ext}")
                with open(path, "wb") as f:
                    f.write(resp.content)
                count += 1
                time.sleep(0.2)  # be polite
    print(f"{name}: {count} images -> {out_dir}")


def verify_images():
    """Delete files OpenCV cannot decode - they would break training."""
    import cv2

    removed = 0
    for root, _dirs, files in os.walk(DATA_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.startswith("."):
                continue
            if cv2.imread(path) is None:
                os.remove(path)
                removed += 1
    print(f"removed {removed} unreadable files")


if __name__ == "__main__":
    for class_name, class_queries in CLASSES.items():
        download_class(class_name, class_queries, IMAGES_PER_CLASS)
    verify_images()
    print("\nNow open data/ and manually delete bad images before training.")
