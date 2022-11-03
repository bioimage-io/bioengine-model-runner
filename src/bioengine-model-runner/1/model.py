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
import time
import asyncio

import uuid
import xarray as xr
import traceback
import logging
import sys

from imjoy_rpc.hypha import RPC
import msgpack
import concurrent.futures

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from ruamel.yaml import YAML
from triton_model_adapter import TritonModelAdapter

yaml = YAML(typ="safe")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("bioengine-runner")
logger.setLevel(logging.INFO)


print(
    f"BioEngine Model Runner (bioimageio.spec: {bioimageio.spec.__version__}, bioimageio.core: {bioimageio.core.__version__})"
)

# The import should be set after the import
import bioimageio.core

executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=10,
)


def get_model_rdf(model_id):
    raw_resource = bioimageio.core.load_raw_resource_description(model_id, update_to_format=="latest")
    return serialize_raw_resource_description_to_dict(raw_resource)


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

        logger.info("Running bioengine model runner...")
    
    def _process_request(self, request, rpc):
        try:
            kwargs = pb_utils.get_input_tensor_by_name(request, "kwargs")
            kwargs = kwargs.as_numpy()
            assert kwargs.dtype == np.object_
            bytes_array = kwargs.astype(np.bytes_).item()
            # Note: This assumes we uses the `imjoy` serialization
            # in the triton client
            data = msgpack.loads(bytes_array)
            kwargs = rpc.decode(data)
            assert "model_id" in kwargs, "model_id must be provided"
            result = {}
            model_id = kwargs["model_id"]
            inputs = kwargs.get("inputs")

            if inputs:
                logger.info("Running model %s...", model_id)
                # weight_format = kwargs.get("weight_format")
                try:
                    start_time = time.time()
                    model_resource = bioimageio.core.load_raw_resource_description(
                        model_id, update_to_format=="latest"
                    )
                    for s in model_resource.inputs:
                        s.root_path =  model_resource.root_path
                    for s in model_resource.outputs:
                        s.root_path =  model_resource.root_path
                    

                    pred_pipeline = create_prediction_pipeline(
                        bioimageio_model=model_resource,
                        model_adapter=TritonModelAdapter(
                            server_url="127.0.0.1:8000",
                            model_id=model_resource.config["bioimageio"]["nickname"],
                            model_version="1",
                            model_resource=model_resource,
                        ),
                    )
                    assert len(model_resource.inputs) == len(
                        inputs
                    ), "inputs length does not match the length in the model definition"
                    input_tensors = [
                        xr.DataArray(
                            input_, dims=tuple(model_resource.inputs[idx].axes)
                        )
                        for idx, input_ in enumerate(inputs)
                    ]
                    output_tensors = pred_pipeline(*input_tensors)
                    output_tensors = [pred.to_numpy() for pred in output_tensors]
                    execution_time = time.time() - start_time
                    result = {
                        "task_id": str(uuid.uuid4()),
                        "outputs": output_tensors,
                        "execution_time": execution_time,
                        "success": True,
                    }
                    logger.info(
                        "Successfully executed model %s, time: %d",
                        model_id,
                        execution_time,
                    )
                except Exception:
                    result = {
                        "error": traceback.format_exc(),
                        "success": False,
                    }
                    logger.info(
                        "Failed to run model %s, error: %s",
                        model_id,
                        traceback.format_exc(),
                    )
            if "return_rdf" in kwargs:
                result["rdf"] = get_model_rdf(model_id)

            data = rpc.encode(result)
            bytes_data = msgpack.dumps(data)
            outputs_bytes_data = np.array([bytes_data], dtype=np.object_)
            out_tensor_1 = pb_utils.Tensor("result", outputs_bytes_data)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_1]
            )
        except Exception:
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[],
                error=pb_utils.TritonError(
                    "An error occurred: " + str(traceback.format_exc())
                ),
            )
        return inference_response

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
        infer_response_awaits = []
        loop = asyncio.get_running_loop()
        rpc = RPC(None, "anon", loop=loop)
        for request in requests:
            _process_request = loop.run_in_executor(executor, self._process_request, request, rpc)
            infer_response_awaits.append(_process_request)
        responses = await asyncio.gather(*infer_response_awaits)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up bioengine runner...")
