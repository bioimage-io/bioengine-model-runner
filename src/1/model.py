# patch for fix "urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]" error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import numpy as np
import xarray as xr
import bioimageio.spec
import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import serialize_raw_resource_description_to_dict
import traceback
import asyncio
import time
import os

import uuid
import xarray as xr
import traceback
import logging
import sys
import torch

from imjoy_rpc.hypha import RPC
import msgpack

import aioprocessing
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

# This is required for pytorch to work with multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("bioengine-runner")
logger.setLevel(logging.INFO)

model_snapshots_directory = os.environ.get("MODEL_SNAPSHOTS_DIRECTORY")
if model_snapshots_directory:
    assert os.path.exists(
        model_snapshots_directory
    ), f'Model snapshots directory "{model_snapshots_directory}" (from env virable MODEL_SNAPSHOTS_DIRECTORY) does not exist'
    MODEL_DIR = os.path.join(model_snapshots_directory, "bioimageio-models")
else:
    MODEL_DIR = os.path.join(
        os.path.dirname(__file__), "../../../bioimageio-models"
    )

os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["BIOIMAGEIO_CACHE_PATH"] = os.environ.get("BIOIMAGEIO_CACHE_PATH", MODEL_DIR)

print(f"BioEngine Model Runner (bioimageio.spec: {bioimageio.spec.__version__}, bioimageio.core: {bioimageio.core.__version__}, MODEL_DIR: {MODEL_DIR})")

# The import should be set after the import
import bioimageio.core

def downlod_model(model_id):
    out_folder = os.path.join(MODEL_DIR, model_id)
    out_path = os.path.join(out_folder, "model.zip")
    if os.path.exists(out_path):
        return out_path
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(model_id, output_path=out_path)
    return model_id

def get_model_rdf(model_id):
    raw_resource = bioimageio.core.load_raw_resource_description(model_id)
    return serialize_raw_resource_description_to_dict(raw_resource)

def start_model_worker(
    model_id, input_queue, output_queue, lock, devices="cpu", weight_format=None
):
    try:
        # out_path = downlod_model(
        #     model_id
        # )
        model_resource = bioimageio.core.load_resource_description(model_id)
        pred_pipeline = create_prediction_pipeline(
            bioimageio_model=model_resource, devices=[devices], weight_format=weight_format
        )
        logger.info("model loaded %s", model_id)
    except Exception:
        output_queue.put(
            {
                "task_id": "start",
                "error": f"Failed to load the model {model_id}: \n" + traceback.format_exc(),
                "success": False,
            }
        )
        return
    else:
        output_queue.put({"task_id": "start", "success": True})

    while True:
        with lock:
            task_info = input_queue.get()
            start_time = time.time()
            if task_info["type"] == "quit":
                output_queue.put({"task_id": "quit", "success": True})
                break
            try:
                assert model_id == task_info["model_id"]
                assert len(model_resource.inputs) == len(
                    task_info["inputs"]
                ), "inputs length does not match the length in the model definition"
                input_tensors = [
                    xr.DataArray(input_, dims=tuple(model_resource.inputs[idx].axes))
                    for idx, input_ in enumerate(task_info["inputs"])
                ]
                output_tensors = pred_pipeline(*input_tensors)
                output_tensors = [pred.to_numpy() for pred in output_tensors]
                execution_time = time.time() - start_time
                output_queue.put(
                    {
                        "task_id": task_info["task_id"],
                        "outputs": output_tensors,
                        "execution_time": execution_time,
                        "success": True,
                    }
                )
                logger.info(
                    "Responded to model execution request (model_id: %s, execution_time: %f)",
                    model_id,
                    execution_time,
                )
            except KeyboardInterrupt:
                logger.info("Terminating by CTRL-C")
                break
            except Exception:
                output_queue.put(
                    {
                        "task_id": task_info["task_id"],
                        "error": traceback.format_exc(),
                        "success": False,
                    }
                )


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])
        self.input_queue = aioprocessing.AioQueue()
        self.output_queue = aioprocessing.AioQueue()
        self.lock = aioprocessing.AioLock()
        self.current_model_signature = None
        gpu_available = torch.cuda.is_available()

        if args["model_instance_kind"] == "GPU" and gpu_available:
            self.devices = "cuda"
        else:
            self.devices = "cpu"
        
        logger.info("Running bioengine model runner using devices: `%s` (GPU available: %s)", self.devices, gpu_available)

    async def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        loop = asyncio.get_running_loop()
        responses = []
        for request in requests:
            try:
                _rpc = RPC(None, "anon")
                kwargs = pb_utils.get_input_tensor_by_name(request, "kwargs")
                kwargs = kwargs.as_numpy()
                assert kwargs.dtype == np.object_
                bytes_array = kwargs.astype(np.bytes_).item()
                # Note: This assumes we uses the `imjoy` serialization
                # in the triton client
                data = msgpack.loads(bytes_array)
                kwargs = _rpc.decode(data)
                assert "model_id" in kwargs, "model_id must be provided"
                result = {}
                model_id = kwargs["model_id"] 
                if "inputs" in kwargs:
                    inputs = kwargs.get("inputs")
                    weight_format = kwargs.get("weight_format")
                    result = await self.execute_model(inputs, model_id, weight_format=weight_format)
                if "return_rdf" in kwargs:
                    result["rdf"] = await loop.run_in_executor(None, get_model_rdf, model_id)
                data = _rpc.encode(result)
                bytes_data = msgpack.dumps(data)
                outputs_bytes_data = np.array([bytes_data], dtype=np.object_)
                out_tensor_1 = pb_utils.Tensor("result", outputs_bytes_data)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_1]
                )
                responses.append(inference_response)
            except Exception:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "An error occurred: " + str(traceback.format_exc())
                        ),
                    )
                )

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
        
    async def load_model(self, model_id, weight_format=None):
        assert not model_id.startswith(
            "http"
        ), "HTTP model url is not allowed, please use Zenodo DOI or nickname."
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, get_model_rdf, model_id)
        except Exception:
            raise KeyError(f"Failed to load the model: {model_id}")

        if self.current_model_signature:
            logger.info("Exiting process (%s)", self.current_model_signature)
            await self.input_queue.coro_put({"type": "quit"})
            await self.output_queue.coro_get()

        logger.info("Starting a new model process for %s", model_id)
        p = aioprocessing.AioProcess(
            target=start_model_worker,
            args=(model_id, self.input_queue, self.output_queue, self.lock),
            kwargs={"devices": self.devices, "weight_format": weight_format},
        )
        p.start()
        # wait for the actual start
        result = await self.output_queue.coro_get()
        assert result["success"] == True, result["error"]
        self.current_model_signature = f"{model_id}:{weight_format}"

    async def get_current_model(self):
        return {"model_id": self.current_model_signature and self.current_model_signature.split(':')[0]}

    async def execute_model(self, inputs, model_id, weight_format=None):
        assert not model_id.startswith(
            "http"
        ), "HTTP model url is not allowed, please use Zenodo DOI or nickname."
        if self.current_model_signature != f"{model_id}:{weight_format}":
            await self.load_model(model_id, weight_format=weight_format)

        task_info = {
            "task_id": str(uuid.uuid4()),
            "type": "model",
            "inputs": inputs,
            "model_id": model_id,
            "devices": self.devices,
            "weight_format": weight_format,
            "reload_model": False,
        }
        await self.input_queue.coro_put(task_info)
        return await self.output_queue.coro_get()

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up bioengine runner...")
