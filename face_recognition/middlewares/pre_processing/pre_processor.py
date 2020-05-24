import pickle

import cv2

from face_recognition.middlewares.detector import detector
from face_recognition.middlewares.pre_processing.align_faces import align_face
from face_recognition.recognizer.recognize import recognize_person


class Person:
    def __init__(self, name):
        self.name = ""


class FaceFrame:
    def set_person(self, person):
        self.person = person

    def get_person(self) -> Person:
        return self.person

    def __init__(self, top_start, top_end, bottom_start, bottom_end, frame):
        self.person = Person("omer")
        self.top_start = top_start
        self.top_end = top_end
        self.bottom_start = bottom_start
        self.bottom_end = bottom_end


def pre_process_frames(frames, ort_session, input_name):
    """
    Pre-Process a list of frames
    :param input_name:
    :param ort_session:
    :param frames: list of frames
    :return: processed list of frames
    """
    return [pre_process_frame(f, ort_session, input_name)[0] for f in frames]


def pre_process_frame(frame, ort_session, input_name):
    """
    Pre-Process a single frame
    :param input_name:
    :param ort_session:
    :param frame: a single image like object
    :return: processed image
    """
    # preprocess img acquired
    frame, boxes = detector.find_face(frame, ort_session, input_name)
    aligned_faces = []
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        aligned_faces.append(align_face(frame, box))

    labels = []
    for i in range(boxes.shape[0]):
        aligned_face = aligned_faces[i]
        labels.append(recognize_person(aligned_face))

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)

        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"face: {labels[i]}"
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

    return frame, aligned_faces


def pre_process_bytes(message, ort_session, input_name):
    frame = pickle.loads(message)
    return pre_process_frame(frame, ort_session, input_name)[0]
