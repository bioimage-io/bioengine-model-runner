import numpy as np
import asyncio
from pyotritonclient import execute

"""

To run this model, run `sh test_triton.sh`
Then, run the server.
"""
async def test_model():
    image = np.random.randint(0, 255, size=(1, 1, 128, 128), dtype=np.uint8).astype(
        "float32"
    )
    kwargs = {"inputs": [image], "model_id": "10.5281/zenodo.5869899", "return_rdf": True}
    ret = await execute(
        [kwargs],
        server_url="http://localhost:8000",
        model_name="bioengine-model-runner", serialization="imjoy"
    )
    result = ret["result"]
    assert "rdf" in result
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )
    print("Test passed")
    
asyncio.run(test_model())