# bioengine-model-runner

A model runner for serving models from bioimage.io, currently all the models are available at https://uk1s3.embassy.ebi.ac.uk/model-repository



## Usage

Go to https://bioimage.io  find a model, then get the model id (e.g.: "10.5281/zenodo.5869899" or "hiding-tiger").

NOTE: Models that contains only weights in `pytorch_state_dict` format are not supported, please convert it to `torchscript`.

### Run inference
In Python you can use the following code to run the model from the BioEngine:

```python
import numpy as np
import asyncio
from pyotritonclient import execute

async def test_model():
    image = np.random.randint(0, 255, size=(1, 1, 128, 128), dtype=np.uint8).astype(
        "float32"
    )
    kwargs = {
        "inputs": [image],
        "model_id": "10.5281/zenodo.5869899",
        "return_rdf": True,
    }
    ret = await execute(
        [kwargs],
        server_url="https://ai.imjoy.io/triton",
        model_name="bioengine-model-runner",
        serialization="imjoy",
    )
    result = ret["result"]
    assert "rdf" in result
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )
    print("Test passed", result["outputs"][0].shape)

asyncio.run(test_model())
```

### Listing models
You can get a list of models from the model repository manifest file in [YAML](https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.yaml) or [JSON](https://raw.githubusercontent.com/bioimage-io/bioengine-model-runner/gh-pages/manifest.bioengine.json).


### Add new models
All the models at bioimage.io will be automatically converted and made available through the BioEngine. The CI in this repo will check for new models twice a day, and convert them to make it available in the BioEngine.

## Development

Here are the steps for generating the conda environment for running the model runner:

```
export PYTHONNOUSERSITE=True # used by conda-pack
conda install -y -c pytorch -c conda-forge bioimageio.core pytorch torchvision cudatoolkit=11.3 cudnn tensorflow onnxruntime xarray
pip install imjoy-rpc aioprocessing
conda install conda-pack
conda-pack
```
