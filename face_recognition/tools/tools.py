import os
from pathlib import Path

import onnxruntime as ort
from cv2 import cv2

# SAVE FRAMES TO DIRECTORY
from face_recognition.middlewares.pre_processing import pre_processor

onnx_path = '../models/ultra_light_640.onnx'


def extract_faces():
    if onnx_path == "":
        exit(1)

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    test_face_image_directory = '../data/before_processing/gt_db'

    # iterate database
    for subdir, dirs, files in os.walk(test_face_image_directory):
        for file in files:
            filename = os.path.join(subdir, file)

            # extract only one image
            if not filename.endswith("s10/07.jpg"):
                continue

            print(filename)
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                #  filter out non-image files
                a = cv2.imread(filename)
                a, b = pre_processor.pre_process_frame(a, ort_session=ort_session, input_name=input_name)
                cv2.imshow("final face", b[0])

                for x in b:
                    if x.size > 0:
                        cv2.imshow("inp", x)
                        filename = filename.replace("data/before_processing/gt_db", "data/gt_db")
                        Path(filename[:len(filename) - 6]).mkdir(parents=True, exist_ok=True)
                        # os.mkdir(filename[:len(filename) - 6])
                        # cv2.imwrite(filename, x)
                        cv2.imshow("tool", x)
                        # cv2.imwrite(filename + "d.jpg", x)
                        cv2.waitKey(0)


if __name__ == '__main__':
    extract_faces()
