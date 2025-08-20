import os
import json
import re
from html import unescape
import unicodedata

INPUT_DIR = "data/CUDA_web_scraped_docs"
OUTPUT_FILE = "data/transformed_web_scraped_docs/cleaned_cuda_docs.json"

def clean_text(text):
    text = unescape(text)  # Decode HTML entities
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace('\xa0', ' ').replace('Â', ' ').replace('ï»¿', '')
    text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
    text = unicodedata.normalize("NFKC", text)
    return text.strip()

all_docs = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                
                if isinstance(doc, list):
                    for item in doc:
                        if isinstance(item, dict):
                            cleaned_content = clean_text(item.get("content", ""))
                            cleaned_title = clean_text(item.get("heading", ""))
                            all_docs.append({
                                "source": filename.replace(".json", ""),
                                "title": cleaned_title,
                                "content": cleaned_content
                            })
                        else:
                            print(f"Skipped non-dict item in {filename}")
                else:
                    print(f"Skipped non-list JSON in {filename}")
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {filename}: {e}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Save cleaned version
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)

print(f"Cleaned and saved {len(all_docs)} documents to {OUTPUT_FILE}")
