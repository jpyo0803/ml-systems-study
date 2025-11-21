import numpy as np
import tritonclient.http as http
import time

client = http.InferenceServerClient(url="localhost:8000")

def send_one():
    x = np.random.randn(1, 4).astype(np.float32)
    inputs = [http.InferInput("INPUT__0", [1,4], "FP32")]
    inputs[0].set_data_from_numpy(x)

    outputs = [http.InferRequestedOutput("OUTPUT__0")]

    resp = client.infer("simple_mlp", inputs=inputs, outputs=outputs)
    return resp.as_numpy("OUTPUT__0")


if __name__ == '__main__':
    while True:
        print(send_one())
        time.sleep(0.1)