import asyncio
import traceback
import imageio
from imjoy_rpc.hypha import connect_to_server
import requests
import numpy as np


SERVERS = [
        "https://ai.imjoy.io/",
        "https://hypha.bioimage.io/",
]


async def main():
    for url in SERVERS:
        print(f"***** Testing {url} *****")
        await test_server(url)


async def test_server(url):
    server = await connect_to_server(
        {"name": "test client", "server_url": url, "method_timeout": 30}
    )

    triton = await server.get_service("triton-client")

    url = "https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.json"

    resp = requests.get(url=url)
    data = resp.json()
    collection = data["collection"]

    for model_info in collection:
        model_id = model_info["id"]
        kwargs = {"model_id": model_id, "inputs": None, "return_rdf": True}
        try:
            ret = await triton.execute(
                inputs=[kwargs],
                model_name="bioengine-model-runner",
                serialization="imjoy"
            )
            assert "result" in ret and "rdf" in ret["result"]
            print("Model test passed: ", ret["result"]["rdf"]["name"], ret["result"]["rdf"].get("id"))
        except:
            print("Model test failed: ", ret["result"]["rdf"]["name"], ret["result"]["rdf"].get("id"))

    # One-off test of the cellpose model
    image = imageio.v3.imread('https://static.imjoy.io/img/img02.png')
    image = image.astype('float32')
    try:
        ret = await triton.execute(
            inputs=[image, {'diameter': 30}],
            model_name="cellpose-python",
            decode_json=True
        )
        # NOTE: Input is RGB, Output is binary mask with leading singleton dimension
        assert ret['mask'].shape[1:] == image.shape[:2], f"Mismatched shapes: {ret['mask'].shape} != {image.shape}"
        print("Model test passed: ", "Cellpose using cellpose-python")
    except:

        print("Model test failed: ", "Cellpose using cellpose-python")
        traceback.print_exc()

    # One-off test of the stardist model
    image = imageio.v3.imread('https://static.imjoy.io/img/img02.png')
    image = image[...,0].astype('uint16')
    try:
        # big_im = (255 * np.random.rand(1000,1000)).astype('uint16')
        # nms_thresh = 100
        # prob_thresh = 0.5
        ret = await triton.execute(
            inputs=[image, {}],
            # inputs=[big_im, {'nms_thresh' : nms_thresh, 'prob_thresh' : prob_thresh}],
            model_name="stardist",
            decode_json=True
        )
        assert ret['mask'].shape == image.shape, f"Mismatched shapes: {ret['mask'].shape} != {image.shape}"
        print("Model test passed: ", "Stardist using stardist")
    except:
        print("Model test failed: ", "Stardist using stardist")
        traceback.print_exc()

asyncio.run(main())
