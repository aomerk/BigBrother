import pickle

import cv2
import numpy as np

from face_recognition.middlewares.detector import detector
from face_recognition.middlewares.pre_processing.align_faces import align_face
from face_recognition.recognizer.recognize import recognize_person

class Person:
    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


def pre_process_frames(frames, ort_session, input_name):
    """
    Pre-Process a list of frames
    :param input_name:
    :param ort_session:
    :param frames: list of frames
    :return: processed list of frames
    """
    return [pre_process_frame(f, ort_session, input_name)[0] for f in frames]


def pre_process_frame(recognizer, frame, ort_session, input_name):
    """
    Pre-Process a single frame
    :param input_name:
    :param ort_session:
    :param frame: a single image like object
    :return frame post processed frame
    :return aligned_faces aligned faces
    """
    # detect faces, original frame and face boxes
    a, boxes = detector.find_face(frame, ort_session, input_name)


    # align facesq
    aligned_faces = []
    boxes[boxes < 0] = 0
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        f = np.copy(frame)
        y = align_face(recognizer, f, box)
        aligned_faces.append(y)

    # recognize person
    labels = []
    for i in range(boxes.shape[0]):
        aligned_face = aligned_faces[i]
        labels.append(recognize_person(recognizer, aligned_face))

    people = []
    for i in range(boxes.shape[0]):
        name, dist = labels[i]
        x1, y1, x2, y2 = boxes[i, :]

        # draw to input frame (for debuging)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        cv2.putText(frame, f"user: {name} {dist}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        people.append(Person(name, x1, y1, x2, y2))

    return people
