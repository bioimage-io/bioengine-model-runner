
import bioimageio.spec
import os
import zipfile
import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
import numpy as np
import xarray as xr

MODEL_DIR = "./models"

model_id = "10.5281/zenodo.5874741"
# bioimageio.core.load_resource_description(model_id)
out_folder = os.path.join(MODEL_DIR, model_id)
out_path = os.path.join(out_folder, "model")
rdf_path = os.path.join(out_path, "rdf.yaml")
if not os.path.exists(rdf_path):
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(model_id, output_path=out_path+".zip")
    with zipfile.ZipFile(out_path+".zip","r") as zip_ref:
        zip_ref.extractall(out_path)
    os.remove(out_path+".zip")

rdf_path = "/data/s3/model-snapshots/bioimageio-models/10.5281/zenodo.5874741/model/rdf.yaml"

devices = "cpu"
weight_format = "torchscript"
model_resource = bioimageio.core.load_resource_description(rdf_path)
pred_pipeline = create_prediction_pipeline(
    bioimageio_model=model_resource, devices=[devices], weight_format=weight_format
)
print("model loaded: ", rdf_path)

input_ = np.zeros([1, 1, 16, 144, 144], dtype=np.uint8)
input_tensors = [
    xr.DataArray(input_, dims=tuple(model_resource.inputs[0].axes))
]
output_tensors = pred_pipeline(*input_tensors)
print(output_tensors)