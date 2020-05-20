import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from face_recognition import face_embedding


def convert_to_embedding(path):
	embs = []
	f_names = os.listdir(path)
	size = (160,160)
	
	for f_name in f_names:
		face = load_img(path + "/" + f_name, target_size = size)
		emb = face_embedding(face)
		embs.append(emb)
	
	np.save("embs" + "_" + path.split("/")[-1] , emb)
	np.save("file_names" + "_" + path.split("/")[-1], f_names)







