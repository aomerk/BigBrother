import argparse
import os
import cv2

# add compilation flags
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of data to find faces from")

args = parser.parse_args()

    #  loop through all test images
    for filename in os.listdir(test_image_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            frame = cv2.imread(test_image_directory + filename)
            cv2.imshow(filename, frame)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)