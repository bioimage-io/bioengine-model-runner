# https://github.com/triton-inference-server/python_backend#business-logic-scripting

from bioimageio.core.prediction_pipeline._model_adapters import ModelAdapter
import xarray as xr
import triton_python_backend_utils as pb_utils
from tritonclient.http import InferenceServerClient

from typing import List, Optional, Sequence


class TritonModelAdapter(ModelAdapter):
    def __init__(self, server_url, model_id, model_version, model_resource):
        self.loaded = True
        self._model_id = model_id
        self._model_version = model_version
        self._model_resource = model_resource
        self._server_url = server_url
        assert not server_url.startswith("http"), "server url should not include schema"
        with InferenceServerClient(self._server_url, verbose=True) as client:
            response = client.get_model_config(self._model_id)
            self._tri_model_info = response

    def _prepare_model(self, bioimageio_model):
        return bioimageio_model

    def _load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        pass

    def _unload(self) -> None:
        pass

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        output_names = [o["name"] for o in self._tri_model_info["output"]]
        input_tensors = [
            pb_utils.Tensor(ip["name"], ip_tensor.to_numpy())
            for ip, ip_tensor in zip(self._tri_model_info["input"], input_tensors)
        ]

        inference_request = pb_utils.InferenceRequest(
            model_name=self._model_id,
            requested_output_names=output_names,
            inputs=input_tensors,
        )
        # all other triton code...
        inference_response = inference_request.exec()
        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            outputs = [
                pb_utils.get_output_tensor_by_name(
                    inference_response, output_name
                ).as_numpy()
                for output_name in output_names
            ]

            outputs = [
                xr.DataArray(output, dims=tuple(ax.axes))
                for output, ax in zip(outputs, self._model_resource.outputs)
            ]
        return outputs
