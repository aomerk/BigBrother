# USAGE
# python align_faces.py --shape-predictor face_model_bigbrother.dat --image images/example_01.jpg
import cv2
import dlib
# import the necessary packages
import imutils
from imutils.face_utils import FaceAligner, rect_to_bb

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../models/face_model_bigbrother.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)


def align_face(image, rects):
    # load the input image, resize it, and convert it to grayscale
    # image = imutils.resize(frame, width=800)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 2)

    # show the original input image and detect faces in the grayscale
    # image

    aligned_faces = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)
        aligned_faces = faceAligned

    return aligned_faces
    # return image[:int(len(image) / 2)]
