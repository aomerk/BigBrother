import argparse
import os

import cv2

# Global Variables
import pre_processing

images = []

# add compilation flags
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of data to find faces from")
args = parser.parse_args()


def show_image(window_name, frame):
    """

    :param window_name: image file name
    :param frame: actual image
    :return:
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

            frame = cv2.imread(dirname + filename)

            # add frame to all images
            images.append(frame)

            # show image
            cv2.imshow(filename, frame)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)


if args.type == "image":
    test_face_image_directory = 'data/images/detection/face/'
    #  loop through all face images
    handle_images_in_directory(test_face_image_directory)

    test_no_face_image_directory = 'data/images/detection/no-face/'
    handle_images_in_directory(test_no_face_image_directory)

    # start pre-processing
    pre_processing.pre_process_frames(images)


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
