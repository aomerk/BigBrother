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

from face_recognition.middlewares.pre_processing.pre_processor import pre_process_frame

class FaceRecognizer:
    def __init__(self, embeddings_path, labels_path, onnx_path, shape_model_path, recon_model, recon_weights):
        self.embeddings = np.load(embeddings_path)
        self.labels = np.load(labels_path)
        self.onnx_model = on.load(onnx_path)
        _ = prepare(self.onnx_model)
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.shape_predictor = dlib.shape_predictor(shape_model_path)
        self.face_aligner = face_utils.facealigner.FaceAligner(self.shape_predictor, desiredFaceWidth=112,
                                                               desiredLeftEye=(0.3, 0.3))
        self.face_net = load_model(recon_model, compile=False)
        self.face_net.load_weights(recon_weights)

    def run(self, frame):
        return pre_process_frame(self, frame, self.ort_session, self.input_name)

if __name__ == "__main__":
	temp = FaceRecognizer(os.getenv("emb_path"), os.getenv("label_path"), os.getenv("onnx_path"), os.getenv("shape_path"), os.getenv("recon_model_path"),os.getenv("recon_weights_path"))
	img = cv2.imread("01.jpg")
	temp.run(img)
