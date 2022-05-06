import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__)+"/../")
import numpy as np
from model import TritonPythonModel

model = TritonPythonModel()
model.initialize({"model_config": "{}", "model_instance_kind": "CPU"})

async def test_execute_model():
    image = np.random.randint(0, 255, size=(1, 1, 128, 128), dtype=np.uint8).astype(
        "float32"
    )
    result = await model.execute_model([image], "10.5281/zenodo.5869899")
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )
    print("Test passed")

asyncio.run(test_execute_model())