import argparse
import os
import cv2

# add compilation flags
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of data to find faces from")

args = parser.parse_args()

# Handle images
if args.type == "image":
    test_image_directory = 'dataset/images/face/'

    #  loop through all test images
    for filename in os.listdir(test_image_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            frame = cv2.imread(test_image_directory + filename)
            cv2.imshow(filename, frame)
            cv2.waitKey(0)
            cv2.destroyWindow(filename)

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
