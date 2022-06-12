from tabnanny import verbose
import numpy as np
import asyncio
from pyotritonclient import execute
import pytest

"""

To run this model, run `sh test_triton.sh`
Then, run the server.
"""

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_model():
    image = np.random.randint(0, 255, size=(1, 1, 128, 128), dtype=np.uint8)
    kwargs = {
        "inputs": [image],
        "model_id": "affable-shark",
        "return_rdf": True,
    }
    ret = await execute(
        [kwargs],
        server_url="http://localhost:5000",
        model_name="bioengine-model-runner",
        serialization="imjoy",
        verbose=True,
    )
    result = ret["result"]
    assert "rdf" in result
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )
    print("Test passed")


# asyncio.run(test_model())
