import numpy as np
import time
from random_request_sampler import RandomRequestSampler

from torchvision import datasets, transforms
import tritonclient.http as http

BATCH_SIZE = 5
LAMBDA = 5  # 초당 평균 5건

class PoissonWorkloadGenerator:
    def __init__(self, sampler, lamda):
        self.sampler = sampler
        self.lamda = lamda # named lamda to avoid keyword
        self.next_time = -1e9

    def generate(self):
        time_to_sleep = self.next_time - time.time()
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        data, label = self.sampler.sample()

        wait_time = np.random.exponential(1 / self.lamda)
        self.next_time = time.time() + wait_time

        return data, label

client = http.InferenceServerClient(url="localhost:8000")
def send_one(data):
    np_data = data.numpy().astype(np.float32)

    inputs = [http.InferInput("INPUT__0", [BATCH_SIZE, 1, 28, 28], "FP32")]
    inputs[0].set_data_from_numpy(np_data)

    outputs = [http.InferRequestedOutput("OUTPUT__0")]

    resp = client.infer("mnist_classifier", inputs=inputs, outputs=outputs)
    return resp.as_numpy("OUTPUT__0")

if __name__ == '__main__':
    # MNIST 데이터셋 준비 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    valid_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

    # Random Sampler
    sampler = RandomRequestSampler(valid_dataset, batch_size=BATCH_SIZE)

    workload_generator = PoissonWorkloadGenerator(sampler, LAMBDA)

    total = 0
    correct_count = 0

    start_time = time.time()
    while True:
        data, label = workload_generator.generate()

        triton_out = send_one(data)

        triton_pred = triton_out.argmax(axis=-1)

        correct_count += (label == triton_pred).sum()
        total += triton_out.shape[0]

        elapsed_time = time.time() - start_time
        print(f"Total: {total}, (# requests / second): {total / BATCH_SIZE / elapsed_time: .2f}, Accuracy: {correct_count / total * 100: .1f}")



