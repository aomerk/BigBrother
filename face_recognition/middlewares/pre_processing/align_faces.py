# USAGE
# python align_faces.py --shape-predictor face_model_bigbrother.dat --image images/example_01.jpg
import cv2
import dlib
# import the necessary packages
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner

shape_predictor = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))


def align_face(image, box):
    # load the input image, resize it, and convert it to grayscale
    # image = imutils.resize(frame, width=800)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 2)
    x1, y1, x2, y2 = box

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # align and resize
    aligned_face = fa.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
    aligned_face = cv2.resize(aligned_face, (112, 112))

    # write to file
    # show the original input image and detect faces in the grayscale
    # image

    return aligned_face
