import numpy as np
from collections import Counter
from global_variables import *
from test_functions import *
import os, sys
from PIL import Image
from predict_classification_net import predict_classification_net

def count_sea_lions(image_name, image):
	corners = np.load(PATH + 'Results/corners_net3_' + image_name + '.npy')
	windows = getWindows(corners, image, 1)
	prediction = predict_classification_net(windows, image_name)
	counts = Counter(prediction)
	np.save(PATH + 'Results/count_'+ image_name + '.npy', counts)
	print(image_name, counts)


def count_folder(path):
	file_names = os.listdir(PATH + path)
	for image_name in file_names:
		if image_name.endswith('.jpg'):
			image = Image.open(PATH + path + image_name)
			count_sea_lions(image_name, image)
        	

"""Testing"""
if __name__ == '__main__':

	try:
		arg1 = sys.argv[1]
	except IndexError:
		print("Command line argument missing. Usage: count_sea_lions.py path_to_folder/ (Test/)")
		sys.exit(1)

	count_folder(arg1)