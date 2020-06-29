import pickle

import cv2


def post_process(recognizer, people) -> bytes:
    message = pickle.dumps(people)
    return message
