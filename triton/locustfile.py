from locust import HttpUser, task, between
import numpy as np

class TritonUser(HttpUser):
    wait_time = between(1, 1)

    @task
    def infer(self):
        x = np.random.randn(10,4).astype(np.float32).tolist()

        body = {
            "inputs":[
                {"name":"INPUT__0", "shape":[10,4], "datatype":"FP32",
                 "data": x}
            ],
            "outputs":[ {"name":"OUTPUT__0"} ]
        }
        self.client.post("/v2/models/simple_mlp/infer", json=body)
