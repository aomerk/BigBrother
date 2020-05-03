import cv2
import numpy as np


def pre_process_frames(frames):
    """
    Pre-Process a list of frames
    :param frames: list of frames
    :return: processed list of frames
    """
    return [pre_process_frame(f) for f in frames]


def pre_process_frame(frame):
    """
    Pre-Process a single frame
    :param frame: a single image like object
    :return: processed image
    """
    # preprocess img acquired
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
    img = cv2.resize(img, (640, 480))  # resize
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    # TODO uncomment for detection
    # img = np.transpose(img, [2, 0, 1])
    # img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img
