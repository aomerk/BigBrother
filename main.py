import argparse
import os
import cv2

# add compilation flags
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of data to find faces from")

args = parser.parse_args()
