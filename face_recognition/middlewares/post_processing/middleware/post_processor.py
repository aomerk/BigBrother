import pickle

import cv2


def post_process(frame) -> bytes:
    message = pickle.dumps(frame)
    return message
