import os
import pickle

import cv2
import onnx as on
import onnxruntime as ort
import zmq
from onnx_tf.backend import prepare

import face_recognizer


def runner():


    #context = zmq.Context()
    #socket = context.socket(zmq.REP)

    #socket.bind("tcp://*:5555")
    cap = cv2.VideoCapture(0)

    recog = face_recognizer.FaceRecognizer(os.getenv("emb_path"), os.getenv("label_path"), os.getenv("onnx_path"), os.getenv("shape_path"), os.getenv("recon_model_path"), os.getenv("recon_weights_path"))
    while True:
        ret, frame = cap.read()
        if frame is not None:
            #  Wait for next request from client
            # message = socket.recv()
            # print("Received request: %s" % message)predictor
            message = pickle.dumps(frame)
            a = recog.run(message)
            print(a)
            #  Do some 'work'
            # time.sleep(0.001)

            #  Send reply back to client
            # socket.send(message)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    runner()
