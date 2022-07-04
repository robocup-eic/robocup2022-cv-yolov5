import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json

image = cv2.imread("data/images/zidane.jpg")
print(image.shape)

host = socket.gethostname()
# host = "192.168.8.2"
port = 10000

c = CustomSocket(host,port)
c.clientConnect()
# print(image.tobytes())
while True : 
    print("Send")
    msg = c.req(image)
    print(msg)
    time.sleep(0.05)