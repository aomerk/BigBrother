import os

import dlib
import numpy as np
import onnx as on
import onnxruntime as ort
import pickle
from imutils import face_utils
from keras.models import load_model
from onnx_tf.backend import prepare
import cv2
import tensorflow as tf

from face_recognition.middlewares.pre_processing.pre_processor import pre_process_frame

class FaceRecognizer:
    def __init__(self, embeddings_path, labels_path, onnx_path, shape_model_path, recon_model, recon_weights):
        self.graph = tf.get_default_graph()
        self.embeddings = np.load(embeddings_path)
        self.labels = np.load(labels_path)
        self.onnx_model = on.load(onnx_path)
        _ = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.shape_predictor = dlib.shape_predictor(shape_model_path)
        self.face_aligner = face_utils.facealigner.FaceAligner(self.shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
        self.face_net = load_model(recon_model, compile=False)
        self.face_net.load_weights(recon_weights)

    def run(self, frame):
        return pre_process_frame(self, frame, self.ort_session, self.input_name)

if __name__ == "__main__":
    embeddings_path="face_recognition/models/embs_database.npy"
    labels_path="face_recognition/models/labels_database.npy"
    onnx_path="face_recognition/models/ultra_light_640.onnx"
    shape_model_path="face_recognition/models/shape_predictor_5_face_landmarks.dat"
    recon_model="face_recognition/models/facenet_keras.h5"
    recon_weights="face_recognition/models/facenet_keras_weights.h5"

    temp = FaceRecognizer(embeddings_path, labels_path, onnx_path, shape_model_path, recon_model, recon_weights)
    img = cv2.imread("01.jpg")
    print("run")
    for res in temp.run(img):
        print(res.name)
        print(f"({res.x1}, {res.y1}, {res.x2}, {res.y2})")
