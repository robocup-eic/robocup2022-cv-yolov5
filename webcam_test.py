import socket
import cv2
import time
from custom_socket import CustomSocket

host = socket.gethostname()
port = 10001

c = CustomSocket(host, port)
c.clientConnect()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

start = time.time()
fps = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame = cv2.resize(frame, (1280, 720))

    end = time.time()
    fps = 1 / (end - start)
    start = end

    cv2.putText(frame, "fps {:.2f}".format(fps), (20,50), cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0), 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    result = c.req(frame)

    print(result)

cap.release()
cv2.destroyAllWindows()
