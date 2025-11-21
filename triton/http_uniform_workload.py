import random
import time
from http_client_random_request import send_one

if __name__ == '__main__':
    while True:
        time.sleep(random.uniform(0.01, 0.5))
        print(send_one())