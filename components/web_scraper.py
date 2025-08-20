import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from tqdm import tqdm

url = "https://docs.nvidia.com/cuda/index.html"

output_dir = "data/CUDA_web_scraped_docs"

#create output dierectory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    """Clean unwanted characters and encoding issues."""
    replacements = {
        "\u00a0": " ",  # non-breaking space
        "Â®": "®",
        "Â": "",
        "ï»¿": "",
        "ï": "",
        "ïƒ": "",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


def get_cuda_doc_links():
    """Get all relevant CUDA documentation links from the main page."""
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    # Get all links inside the main content section
    links = []
    for a_tag in soup.select("a"):
        href = a_tag.get("href")
        text = a_tag.get_text(strip=True)
        if href and (
            href.endswith(".html") or "/cuda-" in href
        ) and not href.startswith("http"):
            full_url = urljoin(url, href)
            links.append((text, full_url))

    # Remove duplicates and sort
    return list(dict.fromkeys(links))


def scrape_doc(url):
    """Scrape one documentation URL and return structured content."""
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        sections = []

        # Grab headings and paragraph content
        for section in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
            heading = clean_text(section.get_text())
            content = []

            # Collect sibling elements until next heading
            for sibling in section.find_next_siblings():
                if sibling.name and sibling.name.startswith("h"):
                    break
                if sibling.name in ["p", "ul", "ol", "pre", "code", "div"]:
                    text = sibling.get_text(separator=" ", strip=True)
                    if text:
                        content.append(clean_text(text))
            if heading and content:
                sections.append({"heading": heading, "content": " ".join(content)})

        return sections
    except Exception as e:
        print(f"[!] Error scraping {url}: {e}")
        return []


def save_to_json(doc_name, data):
    """Save scraped document to a JSON file."""
    safe_name = doc_name.replace(" ", "_").replace("/", "_").lower()
    file_path = os.path.join(output_dir, f"{safe_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("[+] Fetching CUDA documentation links...")
    doc_links = get_cuda_doc_links()

    print(f"[+] Found {len(doc_links)} documents. Starting scrape...\n")

    for doc_name, url in tqdm(doc_links):
        print(f"\n[+] Scraping: {doc_name} -> {url}")
        sections = scrape_doc(url)
        if sections:
            save_to_json(doc_name, sections)
        else:
            print(f"[!] Skipped empty: {doc_name}")

    print(f"\n✅ All documents saved to: {output_dir}")


if __name__ == "__main__":
    main()