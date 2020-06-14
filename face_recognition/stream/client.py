import pickle
import time

import cv2
import zmq

HOST = ''
PORT = 8089


def runner():
    #
    #   Connects REQ socket to tcp://localhost:5555
    #
    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to authentication server")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if frame is not None:
            start = time.time()

            # dump and send frame
            data = pickle.dumps(frame)
            socket.send(data)
            # show reply to client
            message = socket.recv()
            if message is not None:
                frame = pickle.loads(message)
                cv2.imshow('frame', frame)
                end = time.time()
                seconds = end - start
                print("Time taken : {0} seconds".format(seconds))
                fps = 1 / seconds
                print("Estimated frames per second : {0}".format(fps))

        # cv2.destroyWindow('frame')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    runner()
