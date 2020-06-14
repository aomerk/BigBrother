import onnx as on
import onnxruntime as ort
import zmq
from onnx_tf.backend import prepare

from face_recognition.middlewares.post_processing.middleware.post_processor import post_process
from face_recognition.middlewares.pre_processing.pre_processor import pre_process_bytes
from face_recognition.recognizer.recognize import recognize_person


def message_handler(message, ort_session, input_name) -> bytes:
    frame = pre_process_bytes(message, ort_session, input_name)
    message = post_process(frame)
    return message


def runner():
    onnx_path = '../models/ultra_light_640.onnx'
    onnx_model = on.load(onnx_path)
    _ = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    context = zmq.Context()
    socket = context.socket(zmq.REP)

    socket.bind("tcp://*:5555")

    while True:
        #  Wait for next request from client
        message = socket.recv()
        # print("Received request: %s" % message)predictor
        message = message_handler(message, ort_session, input_name)

        #  Do some 'work'
        # time.sleep(0.001)

        #  Send reply back to client
        socket.send(message)


if __name__ == '__main__':
    runner()
