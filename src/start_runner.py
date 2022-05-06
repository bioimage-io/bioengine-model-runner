import asyncio
import time
import os
import argparse
from imjoy_rpc.hypha import connect_to_server

import numpy as np
import bioimageio.core
from bioimageio.core.prediction_pipeline import create_prediction_pipeline
import aioprocessing
import uuid
import xarray as xr
import traceback
import logging
import sys
import torch

logging.basicConfig(
    stream=sys.stdout,
    filename="bioengine-model-runner.log",
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("bioengine-runner")
logger.setLevel(logging.INFO)

DEFAULT_DEVICES = os.environ.get("BIOIMAGEIO_DEFAULT_DEVICES", "cpu")

logger.info("GPU available: %s, Default Devices: %s", torch.cuda.is_available(), DEFAULT_DEVICES)

def downlod_model(model_id, model_dir):
    out_folder = os.path.join(model_dir, model_id)
    out_path = os.path.join(out_folder, "model.zip")
    if os.path.exists(out_path):
        return out_path
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(model_id, output_path=out_path)
    return out_path


def start_model_worker(
    model_id, input_queue, output_queue, lock, devices=DEFAULT_DEVICES, weight_format=None
):
    out_path = downlod_model(
        model_id, os.environ.get("BIOIMAGEIO_MODEL_DIR", "./bioimageio-models")
    )
    model_resource = bioimageio.core.load_resource_description(out_path)
    pred_pipeline = create_prediction_pipeline(
        bioimageio_model=model_resource, devices=[devices], weight_format=weight_format
    )
    logger.info("model loaded %s", model_id)
    while True:
        with lock:
            task_info = input_queue.get()
            start_time = time.time()
            if task_info["type"] == "quit":
                output_queue.put({"success": True})
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
                    "Responding to model execution request (model_id: %s, execution_time: %f)",
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


input_queue = aioprocessing.AioQueue()
output_queue = aioprocessing.AioQueue()
lock = aioprocessing.AioLock()
current_model_id = None


async def load_model(model_id, devices=DEFAULT_DEVICES, weight_format=None):
    global current_model_id
    assert not model_id.startswith(
        "http"
    ), "HTTP model url is not allowed, please use Zenodo DOI or nickname."
    if current_model_id:
        logger.info("Exiting process (%s)", current_model_id)
        await input_queue.coro_put({"type": "quit"})
        await output_queue.coro_get()
    logger.info("Starting a new model process for %s", model_id)
    p = aioprocessing.AioProcess(
        target=start_model_worker,
        args=(model_id, input_queue, output_queue, lock),
        kwargs={"devices": devices, "weight_format": weight_format},
    )
    p.start()
    current_model_id = model_id


async def get_current_model():
    global current_model_id
    return {"model_id": current_model_id}


async def execute_model(inputs, model_id, devices=DEFAULT_DEVICES, weight_format=None):
    global current_model_id
    assert not model_id.startswith(
        "http"
    ), "HTTP model url is not allowed, please use Zenodo DOI or nickname."
    if current_model_id != model_id:
        await load_model(model_id, devices=devices, weight_format=weight_format)

    task_info = {
        "task_id": str(uuid.uuid4()),
        "type": "model",
        "inputs": inputs,
        "model_id": model_id,
        "devices": devices,
        "weight_format": weight_format,
        "reload_model": False,
    }
    await input_queue.coro_put(task_info)
    return await output_queue.coro_get()


async def test_service():
    image = np.random.randint(0, 255, size=(1, 1, 128, 128), dtype=np.uint8).astype(
        "float32"
    )
    result = await execute_model([image], "10.5281/zenodo.5869899")
    assert result["success"] == True, result["error"]
    assert result["outputs"][0].shape == (1, 2, 128, 128), str(
        result["outputs"][0].shape
    )


async def start_service(
    service_id, server_url="https://ai.imjoy.io/", workspace=None, token=None
):
    print(f"Starting bioengine model runner service...")
    client_id = service_id + "-client"
    api = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": server_url,
            "workspace": workspace,
            "token": token,
        }
    )

    await api.register_service(
        {
            "id": service_id,
            "config": {
                "visibility": "public",
            },
            "execute_model": execute_model,
            "load_model": load_model,
            "get_current_model": get_current_model,
        }
    )
    print(
        f"Service (client_id={client_id}) started successfully, available at https://ai.imjoy.io/{api.config.workspace}/services"
    )
    print("Workspace: ", api.config.workspace)
    token = await api.generate_token({"expires_in": 360000})
    print("Connection token: ", token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        type=str,
        default="https://ai.imjoy.io",
        help="URL for the hypha server",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace for connecting to the hypha server",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token for connecting to the hypha server",
    )
    opt = parser.parse_args()
    loop = asyncio.get_event_loop()
    asyncio.run(test_service())
    loop.create_task(
        start_service(
            "bioengine-model-runner",
            server_url=opt.server_url,
            workspace=opt.workspace,
            token=opt.token,
        )
    )
    loop.run_forever()
