import pickle

import cv2


def post_process(people) -> bytes:
    message = pickle.dumps(people)
    return message
