import requests
import json
import os

model_snapshots_directory = os.environ.get("MODEL_SNAPSHOTS_DIRECTORY")
if model_snapshots_directory:
    assert os.path.exists(
        model_snapshots_directory
    ), f'Model snapshots directory "{model_snapshots_directory}" (from env virable MODEL_SNAPSHOTS_DIRECTORY) does not exist'
    MODEL_DIR = os.path.join(model_snapshots_directory, "bioimageio-models")
else:
    MODEL_DIR = os.path.join(
        os.path.dirname(__file__), "../../../bioimageio-models"
    )

os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["BIOIMAGEIO_CACHE_PATH"] = os.environ.get("BIOIMAGEIO_CACHE_PATH", MODEL_DIR)

# The import should be set after the import
import bioimageio.core

if __name__ == "__main__":

    response = requests.get("https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/rdf.json")
    summary = json.loads(response.content)
    collection = summary["collection"]
    for item in collection:
        if item["type"] == "model" and item["id"].startswith("10.5281/zenodo."):
            try:
                bioimageio.core.load_resource_description(item["id"])
                print("Model downloaded: " + item["id"])
            except Exception as exp:
                print(f"Failed to fetch model: {item['id']}, error: {exp}")