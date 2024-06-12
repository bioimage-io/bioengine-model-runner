import requests
import json
import os
import zipfile
from tqdm import tqdm

model_snapshots_directory = os.environ.get("MODEL_SNAPSHOTS_DIRECTORY")
if model_snapshots_directory:
    assert os.path.exists(
        model_snapshots_directory
    ), f'Model snapshots directory "{model_snapshots_directory}" (from env virable MODEL_SNAPSHOTS_DIRECTORY) does not exist'
    MODEL_DIR = os.path.join(model_snapshots_directory, "bioimageio-models")
else:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../bioimageio-models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["BIOIMAGEIO_CACHE_PATH"] = os.environ.get("BIOIMAGEIO_CACHE_PATH", MODEL_DIR)

# The import should be set after the import
import bioimageio.core
from bioimageio.spec import serialize_raw_resource_description_to_dict

TRITON_MODEL_DIR = "./model-repository"

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
                # bioimageio.core.load_resource_description(model_id)
                out_folder = os.path.join(MODEL_DIR, model_id)
                out_path = os.path.join(out_folder, "model")
                rdf_path = os.path.join(out_path, "rdf.yaml")
                if not os.path.exists(rdf_path):
                    os.makedirs(out_folder, exist_ok=True)

                    bioimageio.core.export_resource_package(
                        model_id, output_path=out_path + ".zip"
                    )
                    with zipfile.ZipFile(out_path + ".zip", "r") as zip_ref:
                        zip_ref.extractall(out_path)
                    os.remove(out_path + ".zip")
                    print(f"Model ({item['id']})downloaded: {out_path}")

                raw_resource = bioimageio.core.load_raw_resource_description(rdf_path, update_to_format="latest")
                model_dict = serialize_raw_resource_description_to_dict(raw_resource)

                print(model_dict["weights"].keys())

            except Exception as exp:
                print(f"Failed to fetch model: {item['id']}, error: {exp}")
