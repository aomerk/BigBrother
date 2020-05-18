import argparse
import os

import cv2
import onnx as on
import onnxruntime as ort
from onnx_tf.backend import prepare

# Global Variables
from middleware.pre_processor import pre_process_frames

images = []

# add compilation flags
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of data to find faces from")
args = parser.parse_args()

onnx_path = '/PATH TO MODEL/ultra_light_640.onnx'
onnx_model = on.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name


def show_image(window_name, frame):
    """

    :param window_name: image file name
    :param frame: actual image
    :return:
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def get_images(dirname):
    """
    set all images from directory
    :param dirname: directory to set images from
    :return:
    """
    for tmp_dir in [dirname + 'face/', dirname + 'no-face/']:
        for filename in os.listdir(tmp_dir):
            #  filter out non-image files
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # read frame
                frame = cv2.imread(tmp_dir + filename)
                # add frame to all images
                images.append(frame)


if args.type == "image":
    test_face_image_directory = 'data/images/detection/'
    #  loop through all face images
    get_images(test_face_image_directory)
    # start pre-processing
    images = pre_process_frames(images, ort_session, input_name)
    [show_image('frame', x) for x in images]


#  Handle video input
elif args.type == "video":
    print("not implemented yet.")
    exit(1)

# Handle stream input
elif args.type == "stream":
    print("not implemented yet")
    exit(1)
# Unknown input type
else:
    print("-t flag must be either video or image")
