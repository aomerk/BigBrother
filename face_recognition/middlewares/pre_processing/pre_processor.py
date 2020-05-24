import pickle

from face_recognition.middlewares.detector import detector


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
    img, detected_faces = detector.find_face(frame, ort_session, input_name)


    return img, detected_faces


def pre_process_bytes(message, ort_session, input_name):
    frame = pickle.loads(message)
    return pre_process_frame(frame, ort_session, input_name)[0]
