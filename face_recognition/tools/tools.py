import os

from cv2 import cv2
import onnxruntime as ort


# SAVE FRAMES TO DIRECTORY
from face_recognition.middlewares.pre_processing import pre_processor

onnx_path = '/home/dfirexii/PycharmProjects/BigBrother/face_recognition/models/ultra_light_640.onnx'

def process_db():
    if onnx_path == "":
        exit(1)

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    test_face_image_directory = '/home/dfirexii/PycharmProjects/BigBrother/face_recognition/data/gt_db'

    for subdir, dirs, files in os.walk(test_face_image_directory):
        for filename in files:
            filename = os.path.join(subdir, filename)
            print(filename)
            #  filter out non-image files
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                a = cv2.imread(filename)
                a, b = pre_processor.pre_process_frame(a, ort_session=ort_session, input_name=input_name)
                for x in b:
                    if x.size > 0:
                        # cv2.imshow("inp", x)
                        cv2.imwrite(filename, x)
                        # cv2.waitKey(0)
    # # self.fail()

if __name__ == '__main__':
    process_db()
