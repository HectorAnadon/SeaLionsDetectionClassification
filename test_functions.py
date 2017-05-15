import numpy as np
from usefulFunctions import changeResolution, getWindow
import os
from PIL import Image
import matplotlib.pyplot as plt
from global_variables import *
import pdb

def sliding_window_net_1(image, padding=PADDING_SLIDING_WINDOW, window_size=ORIGINAL_WINDOW_DIM):
	windows = []
	corners = []
	size_x,size_y = image.size
	x = 0
	y = 0

	while (y+window_size <= size_y):

		#crop image
		window = image.crop((x,y,x+window_size,y+window_size))
		#change resolution
		corners.append(np.array([x,y]))
		windows.append(changeResolution(window, 3))
		#update window
		x += padding
		if (x+window_size > size_x):
			if (x+window_size - padding == size_x):
				x = 0
				y += padding
				if (y+window_size > size_y):
					if (y+window_size - padding == size_y):
						break
					elif (y+window_size - padding < size_y):
						y = size_y-window_size
					else:
						break
			elif (x+window_size - padding < size_x):
				x = size_x-window_size
			else:
				x = 0
				y += padding
				if (y+window_size > size_y):
					if (y+window_size - padding == size_y):
						break
					elif (y+window_size - padding < size_y):
						y = size_y-window_size
					else:
						break

	#Normalize
	windows = np.stack(windows) / 255.0

	return windows, np.stack(corners)

def calibrate(corners,predictions,dict):
	assert (len(predictions) == corners.shape[0])
	for i in range(len(predictions)):
		corners[i,0] += dict[predictions[i]][0]
		corners[i,1] += dict[predictions[i]][1]
	return corners

"""Testing"""
if __name__ == "__main__":
	# file_names = os.listdir("Data/Train/")
	# image = Image.open("Data/Train/" + file_names[0])
	# print(file_names[0])
	# size_x,size_y = image.size
	# print(size_x, size_y)
	# windows, corners = sliding_window_net_1(image)
	# print(windows.shape)
	#print(windows[0])
	#print(corners[0])
	#plt.imshow(windows[0])
	#plt.show()
	#plt.imshow(getWindow("Data/Train/" + file_names[0], 0, 0, 3))
	#plt.show()
	#print(corners[79542])
	#plt.imshow(windows[79542])
	#plt.show()
	#print(corners[159083])
	#plt.imshow(windows[159083])
	#plt.show()

	movs = dict()
	m = 0
	for delta_x in X_N:
		for delta_y in Y_N:
			movs[m] = [-delta_x, -delta_y]
			m += 1

	predictions = np.array([1,2,3,4,3,2,5,6,3,2])
	corners = np.zeros([10,2])
	corners2 = calibrate(corners, predictions, movs)