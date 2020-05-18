import cv2
import numpy as np
import glob
from keras.models import load_model
#from keras.models import load_weights
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array

face_net = load_model('facenet_keras.h5')
face_net.load_weights('facenet_keras_weights.h5')
 

def verification(face1, face2, dist_type, l2):
	
	if l2 == 1:
		emb1 = normalizer_l2(face_embedding(face1))
		emb2 = normalizer_l2(face_embedding(face2))
	elif l2 == 0:
		emb1 = face_embedding(face1)
		emb2 = face_embedding(face2)
	else:
		raise '%d is undefined, should be 1 or 0' % l2
	
	if dist_type == "euclidian" :
		threshold = 0.5 
		dist = euclidian_distance(emb1,emb2)
	elif dist_type == "cosine_similarity":
		threshold = 0.07
		dist = cosine_similarity(emb1,emb2)
	else:
		raise '%d is undefined, should be euclidian or cosine_similarity' % dist_type
	
	if dist < threshold:
		return 1
	else:
		return 0


def face_embedding(face):
	face_array = np.asarray(face).astype('float32')
	#standardize for facenet
	standardized = (face_array - face_array.mean()) / face_array.std()
	exp = np.expand_dims(standardized, axis=0)
	emb = face_net.predict(exp)
	return emb[0,:]

def euclidian_distance(emb1, emb2):
	diff = np.subtract(emb1, emb2)
	euclidian = np.sum(np.square(diff))
	return euclidian

def cosine_similarity(emb1, emb2):
	return np.dot(emb1, emb2) / (np.sqrt(np.dot(emb1, emb2)) * np.sqrt(np.dot(emb1, emb2)))


#try with and without normalizer
def normalizer_l2(x):
 return x / np.sqrt(np.sum(np.multiply(x, x)))


#folder_name corresponds to id
def data_set(path):
	faces = []
	labels = []
	size = (160, 160)
	#path = "input_data"
	#there are pictures in input data without any folders with label name
	inputs = glob.glob(path + "/*.jpg")
	for filename in inputs:
		id = filename.split(".")[0].split("/")[1]
		face = load_img(filename, target_size = size)
		faces.append(face)
		labels.append(id)
	return faces, labels

#file_name corresponds to id
def input_data(path):
	faces = []
	labels = []
	#there are pictures in folders, f_name corresponds to id
	size = (160, 160)
	folders = glob.glob(path + "/*")
	for f_name in folders:
		pics = glob.glob(f_name + "/*.jpg")
		for pic_name in pics:
			face = load_img(pic_name, target_size = size)
			faces.append(face)
			labels.append(f_name.split("/")[1])

	return faces, np.array(labels)

def cross_predict(test, train, exp_tst, exp_tr):
	score = 0
	false = []
	true = []
	n = len(test)
	for i, img_test in enumerate(test):
		for j, img_train in enumerate(train):
			ver = verification(img_test, img_train, "euclidian", 1)
			expected = 1 if exp_tst[i] == exp_tr[j] else 0
			result = 1 if ver == expected else 0
			score += result
			
			if result == 0:
				false.append("test: "+ str(exp_tst[i]) + " train: " + str(exp_tr[j]))
			else:
				true.append("test: "+ str(exp_tst[i]) + " train: " + str(exp_tr[j]))
			
	print(expected_tst)
	print(exp_tr)
	
	print("Total pics: " + str(n))
	print("Score: " + str(score))
	print("False predictions: ")
	print(false)
	print("True predictions: ")
	print(true)
	
	return score, false



to_predict, expected_tst = data_set("input_data")
to_compare, expected_tr = input_data("data")
score, falses = cross_predict(to_predict, to_compare, expected_tst, expected_tr)


