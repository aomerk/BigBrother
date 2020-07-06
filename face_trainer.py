import pickle

import cv2
import numpy as np
import os

import numpy as np
from keras.preprocessing.image import load_img

from face_recognizer import FaceRecognizer

from face_recognition.middlewares.detector import detector
from face_recognition.middlewares.pre_processing.align_faces import align_face
from face_recognition.recognizer.recognize import recognize_person

from face_recognition.recognizer.recognize import face_embedding

def get_embeddings_from_raw_image(recognizer, frame):
    """
    Pre-Process a single frame
    :param input_name:
    :param ort_session:
    :param frame: a single image like object
    :return frame post processed frame
    :return aligned_faces aligned faces
    """
    # detect faces, original frame and face boxes
    a, boxes = detector.find_face(frame, recognizer.ort_session, recognizer.input_name)


    # align facesq
    aligned_faces = []
    boxes[boxes < 0] = 0
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        f = np.copy(frame)
        y = align_face(recognizer, f, box)
        aligned_faces.append(y)

    embs = []
    for i in range(boxes.shape[0]):
        aligned_face = aligned_faces[i]
        embs.append(face_embedding(recognizer, aligned_face))

    return embs

if __name__ == "__main__":
    face_recognizer = FaceRecognizer(
        embeddings_path="face_recognition/models/embs_database.npy",
        labels_path="face_recognition/models/labels_database.npy", 
        onnx_path="face_recognition/models/ultra_light_640.onnx",
        shape_model_path="face_recognition/models/shape_predictor_5_face_landmarks.dat",
        recon_model="face_recognition/models/facenet_keras.h5",
        recon_weights="face_recognition/models/facenet_keras_weights.h5")

    embs = np.load("face_recognition/models/embs_database.npy")
    labels = np.load("face_recognition/models/labels_database.npy")

    f_names = os.listdir("faces")
    for f_name in f_names:
        frame = load_img("faces/" + f_name)

        embs = embs + get_embeddings_from_raw_image(face_recognizer, frame)
        labels.append("f_name")

    np.save("face_recognition/models/embs_database.npy", embs)
    np.save("face_recognition/models/labels_database.npy", labels)


