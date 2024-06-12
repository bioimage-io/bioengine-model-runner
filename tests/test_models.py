import numpy as np
from pyotritonclient import execute, get_config
import pytest
import io
import requests
import json
from tqdm import tqdm

from ruamel.yaml import YAML
import urllib

yaml = YAML(typ="safe")


def get_models():
    response = requests.get(
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/collection.json"
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


# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

# TODO: add remaining backends, tensorflow will need conversion
# https://github.com/triton-inference-server/backend/blob/main/README.md#backends
backend_mapping = {
    "torchscript": {"name": "pytorch", "extension": ".pt"},
    "onnx": {"name": "onnxruntime", "extension": ".onnx"},
}


def triton_to_np_dtype(dtype):
    if dtype == "BOOL":
        return bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "BYTES":
        return np.object_
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
        return None, None

    return backend_mapping[selected], weights[selected]["source"]


async def test_all_models():
    for model_summary in get_models():
        model_id = model_summary["id"]
        response = requests.get(model_summary["rdf_source"])
        rdf = yaml.load(response.content)
        nickname = rdf["config"]["bioimageio"]["nickname"]
        print(f"Testing {model_id} ({nickname})...")
        backend_info, _ = get_backend_and_source(rdf["weights"])
        if not backend_info:
            continue

        config = await get_config(
            server_url="http://localhost:5000", model_name=nickname
        )

        input_tesors = []
        for idx, inp in enumerate(rdf["test_inputs"]):
            data = np.load(io.BytesIO(requests.get(inp).content))
            # dtype = rdf['inputs'][idx]['data_type']
            dtype = config["input"][idx]["data_type"]
            dtype = "float32"  # triton_to_np_dtype(dtype.replace('TYPE_', ''))
            input_tesors.append(data.astype(dtype))

        output_tesors = []
        for idx, out in enumerate(rdf["test_outputs"]):
            data = np.load(io.BytesIO(requests.get(out).content))
            # dtype = rdf['outputs'][idx]['data_type']
            dtype = "float32"
            output_tesors.append(data.astype(dtype))

        # run inference
        results = await execute(
            inputs=input_tesors,
            server_url="http://localhost:5000",
            model_name=nickname,
            decode_bytes=True,
        )
        for idx, output in enumerate(output_tesors):
            output_tensor_name = config["output"][idx]["name"]
            np.testing.assert_array_almost_equal(
                output, results[output_tensor_name], decimal=2
            )
        print(f"{model_id} ({nickname}) passed")
