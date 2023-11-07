import asyncio
import traceback
import imageio
from imjoy_rpc.hypha import connect_to_server
import requests


async def main():
    server = await connect_to_server(
        {"name": "test client", "server_url": "https://hypha.bioimage.io/", "method_timeout": 30}
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
    try:
        ret = await triton.execute(
            inputs=[image.astype('float32'), {'diameter': 30}],
            model_name="cellpose-python",
            decode_json=True
        )
        print(ret.keys())
        print(ret)
        mask = ret['mask'][0]
        print("Model test passed: ", "Cellpose usinig cellpose-python")
    except:

        print("Model test failed: ", "Cellpose using cellpose-python")
        traceback.print_exc()

    # One-off test of the stardist model
    image = imageio.v3.imread('https://static.imjoy.io/img/img02.png')
    try:
        ret = await triton.execute(
            inputs=[image.astype('float32'), {'diameter': 30}],
            model_name="stardist-python",
            decode_json=True
        )
        print(ret.keys())
        print(ret)
        mask = ret['mask'][0]
        print("Model test passed: ", "Stardist using stardist-python")
    except:
        print("Model test failed: ", "Cellpose using cellpose-python")
        traceback.print_exc()

asyncio.run(main())
