import time

import numpy as np

from init import embeddings
from init import labels
from init import face_net


def recognize_person(frame):
    """
    Recognize person in frame
    :param frame: only face part of image
    :return: person info? TODO
    """

    # face_embedding array
    # DO STUFF
    face = face_embedding(frame)
    for idx, embed in enumerate(embeddings):
        found, dist = verification(face, embed, "euclidian", 1)
        if found == 1:
            print(labels[idx])
            return labels[idx], dist

    # Return person info + frame ?
    return "unknown", 100


def verification(face1, face2, dist_type, l2):
    if l2 == 1:
        # input: image
        """
        emb1 = normalizer_l2(face_embedding(face1))
        emb2 = normalizer_l2(face_embedding(face2))
        """
        # input: embedding
        emb1 = normalizer_l2(face1)
        emb2 = normalizer_l2(face2)
    elif l2 == 0:
        emb1 = face_embedding(face1)
        emb2 = face_embedding(face2)
    else:
        raise '%d is undefined, should be 1 or 0' % l2

    if dist_type == "euclidian":
        threshold = 0.5
        dist = euclidian_distance(emb1, emb2)
    elif dist_type == "cosine_similarity":
        threshold = 0.07
        dist = cosine_similarity(emb1, emb2)
    else:
        raise '%d is undefined, should be euclidian or cosine_similarity' % dist_type

    if dist < threshold:
        return 1, dist
    else:
        return 0, dist


def face_embedding(face):
    face_array = np.asarray(face).astype('float32')
    # standardize for facenet
    standardized = (face_array - face_array.mean()) / face_array.std()
    exp = np.expand_dims(standardized, axis=0)
    emb = face_net.predict(exp)
    return emb[0, :]


def euclidian_distance(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    euclidian = np.sum(np.square(diff))
    return euclidian


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.sqrt(np.dot(emb1, emb2)) * np.sqrt(np.dot(emb1, emb2)))


# try with and without normalizer
def normalizer_l2(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def cross_predict(test, train, exp_tst, exp_tr, pics):
    score = 0
    false = []

    for i, img_test in enumerate(test):
        s_time = time.time()

        for j, img_train in enumerate(train):
            ver, dist = verification(img_test, img_train, "euclidian", 1)
            expected = (exp_tst[i] == exp_tr[j])
            result = (ver == expected)
            score += result
            if result == 0:
                false.append(
                    "test: " + str(exp_tst[i]) + "/" + str(pics[i]) + " train:" + str(exp_tr[j]) + " : " + str(dist))

    return score, false
