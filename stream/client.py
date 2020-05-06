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

    #  Do 10 requests, waiting each time for a response
    for request in range(1):
        frame = cv2.imread("1.jpeg")
        data = pickle.dumps(frame)
        print("Sending request %s …" % data)
        socket.send(data)

        #  Get the reply.
        message = socket.recv()
        frame = pickle.loads(message)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyWindow('frame')
        print("Received reply %s [ %s ]" % (request, message))


if __name__ == '__main__':
    runner()
