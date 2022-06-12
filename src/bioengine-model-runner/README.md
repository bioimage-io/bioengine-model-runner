
## Creating the environment

```
export PYTHONNOUSERSITE=True
conda create -n bioengine-model-runner python=3.8 # we use 3.8 so we don't need to provide the stub
pip install -U bioimageio.core bioimageio.spec xarray imjoy-rpc requests msgpack tritonclient[http]
conda install conda-pack
conda-pack
```

