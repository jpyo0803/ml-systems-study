import time
from http_client_random_request import send_one


if __name__ == '__main__':
    while True:
        # burst
        for _ in range(20):
            print(send_one())
        time.sleep(2)