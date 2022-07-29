
## Creating the environment

```
export PYTHONNOUSERSITE=True
conda create -n bioengine-model-runner python=3.8 # we use 3.8 so we don't need to provide the stub
pip install -U https://github.com/bioimage-io/core-bioimage-io-python/archive/e335077dfaa282fb4982b85ef17cb46ddd939837.zip
conda install -c conda-forge cupy # This is needed for now, for converting GPU tensor to numpy, see https://github.com/triton-inference-server/server/issues/3992#issuecomment-1055464670
pip install -U bioimageio.spec xarray imjoy-rpc requests msgpack tritonclient[http]
conda install conda-pack
conda-pack
```


## Local testing
```
docker-compose up
```

```
cp -r src/bioengine-model-runner/ models/
python src/bioengine-model-runner/1/tests/test_model_triton.py
```

