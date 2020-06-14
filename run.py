# Global Variables
from face_recognition.middlewares.post_processing.middleware.post_processor import post_process
from face_recognition.middlewares.pre_processing.pre_processor import pre_process_bytes
from init import ort_session, input_name


def run(message) -> bytes:
    frame = pre_process_bytes(message, ort_session, input_name)
    message = post_process(frame)
    return message
