import pickle

import cv2
import zmq

HOST = ''
PORT = 8089


def runner():
    #
    #   Hello World client in Python
    #   Connects REQ socket to tcp://localhost:5555
    #   Sends "Hello" to server, expects "World" back
    #
    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to hello world server…")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if frame is not None:
            data = pickle.dumps(frame)
            socket.send(data)

            #  Get the reply.
            message = socket.recv()
            frame = pickle.loads(message)
            cv2.imshow('frame', frame)
            # cv2.destroyWindow('frame')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    runner()
