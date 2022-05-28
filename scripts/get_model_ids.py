import requests
import json
import os
import zipfile
from tqdm import tqdm


if __name__ == "__main__":

    response = requests.get("https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/rdf.json")
    summary = json.loads(response.content)
    collection = summary["collection"]
    for item in tqdm(collection):
        if item["type"] == "model" and item["id"].startswith("10.5281/zenodo."):
            try:
                model_id = item["id"]
                print(f' - id: {model_id}')
            except Exception as exp:
                print(f"Failed to fetch model: {item['id']}, error: {exp}")