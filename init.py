import os

import dlib
import numpy as np
import onnx as on
import onnxruntime as ort
from imutils import face_utils
from keras.models import load_model
from onnx_tf.backend import prepare

# from keras.models import load_weights

embeddings_path = os.environ.get("EMBEDDINGS_PATH")
embeddings = np.load(embeddings_path)

labels_path = os.environ.get("LABELS_PATH")
labels = np.load("/home/dfirexii/PycharmProjects/BigBrother/face_recognition/data/labels_data.npy")

# detect faces

onnx_path = os.environ.get('ONNX_MODEL')
onnx_model = on.load(onnx_path)
_ = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# align faces -------
# initialize dlib  face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
shape_model_path = os.environ.get('SHAPE_MODEL')
shape_predictor = dlib.shape_predictor(shape_model_path)
face_aligner = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

# Recognizing tools
recon_model = os.environ.get('RECON_MODEL')
recon_weights = os.environ.get('RECON_WEIGHTS')
face_net = load_model(recon_model, compile=False)
face_net.load_weights(recon_weights)


# initializer
def init():
    print("init")
