import numpy as np
import time
from http_client_random_request import send_one

LAMBDA = 5  # 초당 평균 5건

if __name__ == '__main__':
    while True:
        wait = np.random.exponential(1/LAMBDA)
        time.sleep(wait)
        print(send_one())