import sys
import os
import asyncio

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__) + "/../")
from model import TritonPythonModel

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    model = TritonPythonModel()
    model.initialize({"model_config": "{}", "model_instance_kind": "CPU"})
    loop = asyncio.get_event_loop()
    loop.run_forever()
