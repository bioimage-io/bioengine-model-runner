import asyncio
import traceback
import os
import sys

import imageio
import requests
import numpy as np
from imjoy_rpc.hypha import connect_to_server


SERVERS = [
    "https://ai.imjoy.io/",
    "https://hypha.bioimage.io/",
]


async def main():
    for url in SERVERS:
        print(f"***** Testing {url} *****")
        errors = await test_server(url)


async def test_server(server_url):
    server = await connect_to_server(
        {"name": "test client", "server_url": server_url, "method_timeout": 30}
    )

    triton = await server.get_service("triton-client")

    url = "https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.json"

    resp = requests.get(url=url)
    data = resp.json()
    collection = data["collection"]
    errors = []

    for model_info in collection:
        model_id = model_info["id"]
        kwargs = {"model_id": model_id, "inputs": None, "return_rdf": True}
        try:
            ret = await triton.execute(
                inputs=[kwargs],
                model_name="bioengine-model-runner",
                serialization="imjoy",
            )
            assert "result" in ret and "rdf" in ret["result"]
            print(
                "Model test passed: ",
                ret["result"]["rdf"]["name"],
                ret["result"]["rdf"].get("id"),
            )
        except:
            print(
                "Model test failed: ",
                ret["result"]["rdf"]["name"],
                ret["result"]["rdf"].get("id"),
            )
            errors.append(
                " ".join(
                    "Model test failed: ",
                    ret["result"]["rdf"]["name"],
                    ret["result"]["rdf"].get("id"),
                )
            )
            errors.append(traceback.format_exc())

    # One-off test of the cellpose model
    image = imageio.v3.imread("https://static.imjoy.io/img/img02.png")
    image = image.astype("float32")
    try:
        ret = await triton.execute(
            inputs=[image, {"diameter": 30}],
            model_name="cellpose-python",
            decode_json=True,
        )
        # NOTE: Input is RGB, Output is binary mask with leading singleton dimension
        assert (
            ret["mask"].shape[1:] == image.shape[:1]
        ), f"Mismatched shapes: {ret['mask'].shape} != {image.shape}"
        print("Model test passed: ", "Cellpose using cellpose-python")
    except:
        print("Model test failed: ", "Cellpose using cellpose-python")
        traceback.print_exc()
        errors.append("Model test failed: Cellpose using cellpose-python")
        errors.append(traceback.format_exc())

    # One-off test of the stardist model
    image = imageio.v3.imread("https://static.imjoy.io/img/img02.png")
    image = image[..., 0].astype("uint16")
    try:
        # big_im = (255 * np.random.rand(1000,1000)).astype('uint16')
        # nms_thresh = 100
        # prob_thresh = 0.5
        ret = await triton.execute(
            inputs=[image, {}],
            # inputs=[big_im, {'nms_thresh' : nms_thresh, 'prob_thresh' : prob_thresh}],
            model_name="stardist",
            decode_json=True,
        )
        assert (
            ret["mask"].shape == image.shape
        ), f"Mismatched shapes: {ret['mask'].shape} != {image.shape}"
        print("Model test passed: ", "Stardist using stardist")
    except:
        print("Model test failed: ", "Stardist using stardist")
        traceback.print_exc()
        errors.append("Model test failed: Stardist using stardist")
        errors.append(traceback.format_exc())

    if errors:
        broadcast_errors(errors, server_url)


def broadcast_errors(errors, server_url):
    try:
        url = os.getenv("AICELL_LAB_SLACK_WEBHOOK_URL")
        if url is None:
            print("AICELL_LAB_SLACK_WEBHOOK_URL not set, can not send errors")
            return
        headers = {"Content-type": "application/json"}
        requests.post(
            url,
            headers=headers,
            json={"text": f"Error(s) occurred in test-triton-models.py on {server_url}:"},
        )
        for error in errors:
            data = {"text": error}
            requests.post(url, headers=headers, json=data)
        requests.post(url, headers=headers, json={"text": "End of errors"})
    except:
        print("Could not send error report to AiCell-Lab 😢")
        traceback.print_exc()

asyncio.run(main())
