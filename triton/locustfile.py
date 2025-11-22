from locust import HttpUser, task, between
from random_request_sampler import RandomRequestSampler
from torchvision import datasets, transforms
import numpy as np

BATCH_SIZE = 5

# MNIST 데이터셋 준비 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
valid_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

# Random Sampler
sampler = RandomRequestSampler(valid_dataset, batch_size=BATCH_SIZE)

class TritonUser(HttpUser):
    wait_time = between(1, 1)

    @task
    def infer(self):
        data, label = sampler.sample()

        data = data.numpy().tolist()

        body = {
            "inputs":[
                {
                    "name":"INPUT__0",
                    "shape":[BATCH_SIZE, 1, 28, 28],
                    "datatype":"FP32",
                    "data": data
                }
            ],
            "outputs":[ {"name":"OUTPUT__0"} ]
        }

        self.client.post("/v2/models/mnist_classifier/infer", json=body)
