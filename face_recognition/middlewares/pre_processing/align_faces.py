import cv2
import dlib

def align_face(recognizer, image, box):
    # load the input image, resize it, and convert it to grayscale

    # image = imutils.resize(image, width=800)
    # rects = detector(gray, 2)
    x1, y1, x2, y2 = box

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # align and resize
    aligned_face = recognizer.face_aligner.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
    aligned_face = cv2.resize(aligned_face, (160, 160))

    return aligned_face
