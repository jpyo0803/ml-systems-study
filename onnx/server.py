import uvicorn
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

import threading
import time
from concurrent.futures import Future
from collections import deque

app = FastAPI()

'''
    CPU 사용시: CPUExecutionProvider
    GPU 사용시: CUDAExecutionProvider
'''
session = ort.InferenceSession('mnist_classifier.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

MAX_BATCH = 16          # dynamic batching 크기
MAX_WAIT_MS = 5000        # 최대 대기 시간

queue = deque()
queue_lock = threading.Lock()

def dynamic_batch_worker():
    while True:
        start = time.time()
        items = []

        # 1) 요청이 충분히 쌓일때까지 대기 
        while len(items) < MAX_BATCH:
            with queue_lock:
                # 큐에 요청이 존재하면 옮기기 
                if queue:
                    item = queue.popleft()
                    items.append(item)
            
            # 2) 시간 초과되면 break
            if (time.time() - start) * 1000 > MAX_WAIT_MS:
                break

        # 만약 요청이 없으면 다시 대기 모드 진입 
        if not items:
            continue

        # 요청 합치기 
        batch_inputs = np.concatenate([it["data"] for it in items], axis=0)
        batch_sizes = [it["batch_size"] for it in items]

        # 모델 실행
        batch_outputs = session.run([output_name], {input_name: batch_inputs})[0]

        # 결과 분할
        idx = 0
        for it, bsz in zip(items, batch_sizes):
            result = batch_outputs[idx:idx + bsz]
            idx += bsz
            it["future"].set_result(result)

threading.Thread(target=dynamic_batch_worker, daemon=True).start()

class MNISTInput(BaseModel):
    data: list # shape: [Batch size, 1, 28, 28]

@app.post('/infer')
def infer_async(body: MNISTInput):
    np_data = np.array(body.data, dtype=np.float32)  # list -> numpy

    future = Future()
    item = {
        "data": np_data,
        "future": future,
        "batch_size": np_data.shape[0]
    }

    with queue_lock:
        queue.append(item)

    # worker가 future.set_result(...) 하기까지 기다림
    result = future.result()

    # numpy -> list로 변환 후 return
    return {"output": result.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)