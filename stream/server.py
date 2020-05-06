import pre_processing
import pickle
import time

import cv2
import zmq


#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#


def runner():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        #  Wait for next request from client
        message = socket.recv()
        # print("Received request: %s" % message)
        frame = pickle.loads(message)
        frame = pre_processing.pre_process_frame(frame)
        message = pickle.dumps(frame)
        #  Do some 'work'
        time.sleep(1)

        #  Send reply back to client
        socket.send(message)


if __name__ == '__main__':
    runner()
