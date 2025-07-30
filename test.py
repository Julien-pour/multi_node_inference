import copy
from tenacity import retry, wait_exponential, wait_random
from concurrent.futures import ThreadPoolExecutor

import subprocess
import numpy as np
from openai import OpenAI, AzureOpenAI
import requests
from requests.exceptions import RequestException
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="qwen/qwen2.5-0.5b-instruct")
parser.add_argument("--port", type=int, default=30000)
args = parser.parse_args()

print("====="*20)
print("====="*20)
print("====="*20)
print("====="*20)
print("====="*20)


print("launching test script for model:", args.model_path)

print("====="*20)
print("====="*20)
print("====="*20)
print("====="*20)
print("====="*20)
def check_server_run(model_path, port=30000):
    """Check if the server is running and serving the correct model.
    Needed when launching multiple inference processes on a cluster.
    """
    try:
        wait_for_server(f"http://127.0.0.1:{port}")
        time.sleep(15)

        req= f"http://127.0.0.1:{port}/get_model_info"
        response = requests.get(
            req,
            headers={"Authorization": "Bearer None"},
        )
        model_id_serv = response.json()["model_path"]
        good_model = response.json()["model_path"] == model_path
        print("model_id_serv", model_id_serv)
        print("model_path", model_path)
        good_model = model_id_serv == model_path
        print("good_model", good_model)
        if not good_model:
            raise Exception("wrong model")
    except:
        return False

    return True


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)

                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)



is_running = check_server_run(args.model_path, args.port)
if not is_running:
    for i in range(4):
        print(f"Server not running, attempt {i+1}/4")
        is_running = check_server_run(args.model_path, args.port)
        if is_running:
            print('=============='*20)
            print("============= Server is running =============")
            print('=============='*20)
            print("Server started successfully.")
            break
        time.sleep(5)

print("try to send request to the server")
client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="None")

response = client.chat.completions.create(
    model=args.model_path,
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(f"Response: {response}")