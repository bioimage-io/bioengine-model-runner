import requests
import json
from tqdm import tqdm


if __name__ == "__main__":
    response = requests.get(
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/collection.json"
    )
    summary = json.loads(response.content)
    collection = summary["collection"]
    for item in tqdm(collection):
        if item["type"] == "model" and item["id"].startswith("10.5281/zenodo."):
            try:
                model_id = item["id"]
                print(f" - id: {model_id}")
            except Exception as exp:
                print(f"Failed to fetch model: {item['id']}, error: {exp}")
