import asyncio
from imjoy_rpc.hypha import connect_to_server
import requests

async def main():
    server = await connect_to_server(
        {"name": "test client", "server_url": "https://ai.imjoy.io", "method_timeout": 30}
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

asyncio.run(main())