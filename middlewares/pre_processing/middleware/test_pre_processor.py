import os
from unittest import TestCase

import cv2
import onnxruntime as ort

from middlewares.pre_processing.middleware.pre_processor import pre_process_frame


class Test(TestCase):

    def test_pre_process_frame(self):
        # SAVE FRAMES TO DIRECTORY
        onnx_path = 'PATH TO ONNX MODEL'
        if onnx_path == "":
            self.fail()

        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        test_face_image_directory = 'PATH TO DIRECTORY'

        for subdir, dirs, files in os.walk(test_face_image_directory):
            for filename in files:
                filename = os.path.join(subdir, filename)
                print(filename)
                #  filter out non-image files
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    a = cv2.imread(filename)
                    a, b = pre_process_frame(a, ort_session=ort_session, input_name=input_name)
                    for x in b:
                        # cv2.imshow("inp", x)
                        cv2.imwrite(filename, x)
                        # cv2.waitKey(0)
    # # self.fail()
