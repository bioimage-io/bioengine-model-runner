import numpy as np
import xarray as xr
import bioimageio.spec
import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
from bioimageio.spec import serialize_raw_resource_description_to_dict
from typing import List, Optional, Sequence
from bioimageio.core.prediction_pipeline._model_adapters import ModelAdapter
from bioimageio.spec.shared.node_transformer import UriNodeTransformer

from bioimageio.core.resource_io import nodes
from bioimageio.core.resource_io.utils import (
    RawNodeTypeTransformer,
    SourceNodeTransformer,
)


class TritonModelAdapter(ModelAdapter):
    def __init__(self, server_url, model_id, model_version, model_resource):
        self.loaded = True
        self._model_id = model_id
        self._model_version = model_version
        self._model_resource = model_resource
        self._server_url = server_url
        assert not server_url.startswith("http"), "server url should not include schema"

    def _load(self, *, devices: Optional[Sequence[str]] = None) -> None:
        pass

    def _unload(self) -> None:
        pass

    def _forward(self, *input_tensors: xr.DataArray) -> List[xr.DataArray]:
        return input_tensors


model_id = "10.5281/zenodo.5874741"
model_resource = bioimageio.core.load_raw_resource_description(model_id, update_to_format="latest")

rd = model_resource
# rd = UriNodeTransformer(root_path=rd.root_path, uri_only_if_in_package=True).transform(rd)
rd = SourceNodeTransformer().transform(rd)
rd = RawNodeTypeTransformer(nodes).transform(rd)

# resolved_data = serialize_raw_resource_description_to_dict(model_resource)
resolved_data = nodes.Model(**rd)
pred_pipeline = create_prediction_pipeline(
    bioimageio_model=resolved_data,
    model_adapter=TritonModelAdapter(
        server_url="127.0.0.1:8000",
        model_id=model_id,
        model_version="1",
        model_resource=model_resource,
    ),
)
