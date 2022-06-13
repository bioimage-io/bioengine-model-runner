from genericpath import exists
import os
import requests
import json
from tqdm import tqdm
from pathlib import Path
import jinja2
import numpy as np
from ruamel.yaml import YAML
import urllib
import zipfile

yaml = YAML(typ="safe")

# TODO: add remaining backends, tensorflow will need conversion
# https://github.com/triton-inference-server/backend/blob/main/README.md#backends
backend_mapping = {
    "torchscript": {"name": "pytorch", "platform": "pytorch_libtorch", "extension": ".pt"},
    "onnx": {"name": "onnxruntime", "platform": "onnxruntime_onnx", "extension": ".onnx"},
    "tensorflow_saved_model_bundle": {"name": "tensorflow", "platform": "tensorflow_savedmodel", "extension": ".savedmodel"},
}

MODELS_DIR = Path("./models")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", leave=False, unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def np_to_triton_dtype(np_dtype):
    if np_dtype == bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


def get_backend_and_source(weights):
    formats = list(weights.keys())
    supported = list(backend_mapping.keys())
    selected = None
    for format in formats:
        if format in supported:
            selected = format
            break
    if not selected:
        return None, None, None

    return selected, backend_mapping[selected], weights[selected]["source"]


def get_models():
    response = requests.get(
        "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/rdf.json"
    )
    summary = json.loads(response.content)
    collection = summary["collection"]
    models = []
    for item in tqdm(collection):
        if item["type"] == "model" and item["id"].startswith("10.5281/zenodo."):
            try:
                model_id = item["id"]
                print(f" - id: {model_id}")
            except Exception as exp:
                print(f"Failed to fetch model: {item['id']}, error: {exp}")
                continue

            models.append(item)
    return models


if __name__ == "__main__":
    template = jinja2.Template(
        Path("./scripts/config_template.pbtxt").read_text(encoding="utf-8")
    )
    model_summaries = get_models()
    for model_summary in model_summaries:
        model_id = model_summary["id"]
        response = requests.get(model_summary["rdf_source"])
        rdf = yaml.load(response.content)
        assert all(
            ["b" in input_["axes"] for input_ in rdf["inputs"]]
        ), "b should always exist in the inputs"
        assert all(
            ["b" in input_["axes"] for input_ in rdf["outputs"]]
        ), "b should always exist in the inputs"

        selected_weight_format, backend_info, weight_source = get_backend_and_source(rdf["weights"])
        if backend_info:
            inputs = [
                {
                    "name": f'{input_["name"]}__{idx}'
                    if backend_info["name"] == "pytorch"
                    else input_["name"],
                    "dtype": "TYPE_FP32",
                    # + np_to_triton_dtype(np.dtype(input_["data_type"])),
                    "dims": [-1 for dim in input_["axes"] if dim != "b"],
                }
                for idx, input_ in enumerate(rdf["inputs"])
            ]
            outputs = [
                {
                    "name": f'{output_["name"]}__{idx}'
                    if backend_info["name"] == "pytorch"
                    else output_["name"],
                    "dtype": "TYPE_"
                    + np_to_triton_dtype(np.dtype(output_["data_type"])),
                    "dims": [-1 for dim in output_["axes"] if dim != "b"],
                }
                for idx, output_ in enumerate(rdf["outputs"])
            ]

            batch_size = 1 if "b" in rdf["inputs"][0]["axes"] else 0
            nickname = rdf["config"]["bioimageio"]["nickname"]
            model_dir = MODELS_DIR / nickname
            model_dir.mkdir(parents=True, exist_ok=True)
            # TensorRT, TensorFlow saved-model, and ONNX models do not require a model configuration file because Triton can derive all the required settings automatically
            if backend_info["name"] in ["tensorflow", "onnxruntime"]:
                include_io_definition = False
            else:
                include_io_definition = True
            config_pbtxt = template.render(
                model_name=nickname,
                backend=backend_info["name"],
                platform=backend_info["platform"],
                batch_size=batch_size,
                inputs=inputs,
                outputs=outputs,
                include_io_definition=include_io_definition
            )
            (model_dir / "config.pbtxt").write_text(config_pbtxt)
            version_dir = model_dir / "1"
            version_dir.mkdir(parents=True, exist_ok=True)
            weight_path = version_dir / f"model{backend_info['extension']}"
            print(f"Processing {nickname}...")
            # unzip the tensorflow bundle
            if selected_weight_format == "tensorflow_saved_model_bundle":
                weight_path.mkdir(exist_ok=True)
                download_url(weight_source, str(weight_path)+".zip")
                with zipfile.ZipFile(str(weight_path)+".zip", 'r') as zip_ref:
                    zip_ref.extractall(weight_path)
                os.remove(str(weight_path)+".zip")
            else:
                download_url(weight_source, weight_path)
            
        else:
            print(f"Skipping model without supported weight format: {rdf['id']}")
