# 1. Download https://uk1s3.embassy.ebi.ac.uk/model-repository/cellpose-model-triton.zip
# And unzip it so you get a models folder (assuming /home/models)
# 2. Download https://uk1s3.embassy.ebi.ac.uk/model-repository/cellpose-triton-gpu.tar.gz
# Do not unzip it but move it to the models folder so you get `cellpose-triton-gpu.tar.gz` and `cellpose-python` under /home/models folder
# 3. Then cd to /home and run: `mkdir trained-models` (so you have also /home/trained-models)
# Then run the following command to start the server:
docker run --gpus '"device=2"' --env MODEL_SNAPSHOTS_DIRECTORY=/trained-models --shm-size=1g --ulimit memlock=-1 -p 5000:8000 -p 5001:8001 -p 5002:8002 -v `pwd`/models:/models -v `pwd`/trained-models:/trained-models --ulimit stack=67108864 -ti --entrypoint=/bin/sh ghcr.io/imjoy-team/triton:22.04-py3 -c "apt update -yq && apt install libgl1-mesa-glx -y && tritonserver --model-repository=/models --strict-model-config=false --log-verbose=0 --model-control-mode=poll --repository-poll-secs=10"
# 4. To test whether the server works, go to http://localhost:5000/v2/models/cellpose-python/versions/1 and you should see the following:
# {"name":"cellpose-python","versions":["1"],"platform":"python","inputs":[{"name":"image","datatype":"FP32","shape":[-1,-1,-1]},{"name":"param","datatype":"BYTES","shape":[1]}],"outputs":[{"name":"mask","datatype":"UINT16","shape":[-1,-1,-1]},{"name":"info","datatype":"BYTES","shape":[1]}]}
# Note: it will take some time for the server to start (e.g. 20s-2minutes), you may also change the gpu device number to the actual one available on your machine.

