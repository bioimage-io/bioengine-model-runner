# bioengine-model-runner

A model runner for serving models from bioimage.io, currently all the models are available at https://uk1s3.embassy.ebi.ac.uk/model-repository



## Development

Here are the steps for generating the conda environment for running the model runner:

```
export PYTHONNOUSERSITE=True # used by conda-pack
conda install -y -c pytorch -c conda-forge bioimageio.core pytorch torchvision cudatoolkit=11.3 cudnn tensorflow onnxruntime xarray
pip install imjoy-rpc aioprocessing
conda install conda-pack
conda-pack
```
