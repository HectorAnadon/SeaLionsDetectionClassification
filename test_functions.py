import numpy as np
from usefulFunctions import cropAndChangeResolution
import os
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from global_variables import *
from NMS import non_max_suppression_slow
import pdb
from make_datasets import save_to_disk

def sliding_window_net_1(image, imageDotted=None, padding=PADDING_SLIDING_WINDOW, window_size=ORIGINAL_WINDOW_DIM):
	windows = []
	corners = []
	size_x,size_y = image.size
	x = 0
	y = 0
	while (y+window_size <= size_y):
		doCrop = True

		if (imageDotted):
			isBlack = cropAndChangeResolution(imageDotted, 'image_name', x, y, window_size, 1)
			pixels = isBlack.getdata()          # get the pixels as a flattened sequence
			nblack = 0
			for pixel in pixels:
			    if pixel == (0,0,0):
			        nblack += 1
			n = len(pixels)
			if (nblack / float(n)) > 0.4:
			    doCrop = False

		if (doCrop):
			window = cropAndChangeResolution(image, 'image_name', x, y, window_size, 3)
			windows.append(np.array(window))
			corners.append(np.array([x,y]))
		
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

	return np.stack(windows) /255.0, np.stack(corners)

def generate_testing_dataset(path, pathToTestFolder):
	file_names = os.listdir(path + pathToTestFolder)

	for image_name in file_names:
		if image_name.endswith('.jpg'):
			print("Processing ", image_name)
			image = Image.open(path + pathToTestFolder + image_name)
			windows, corners = sliding_window_net_1(image)
			save_to_disk(windows, corners, path + 'TestDatasets/sliding_window_' + image_name +'.h5')

# Returns a window given the path, x and y
def getWindows(corners, image, resolution_lvl):
	windows = []
	for c in corners:
		windows.append(cropAndChangeResolution(image,'',c[0],c[1],ORIGINAL_WINDOW_DIM,resolution_lvl))
	return np.stack(windows) / 255.0

def calibrate(corners,predictions,dict):
	assert (len(predictions) == corners.shape[0])
	for i in range(len(predictions)):
		if len(predictions[i]) == 0:
			continue
		elif len(predictions[i]) == 1:
			corners[i,0] += dict[predictions[i][0]][0]
			corners[i,1] += dict[predictions[i][0]][1]
		else:
			mov_x = 0
			mov_y = 0
			for p in predictions[i]:
				mov_x += dict[p][0]
				mov_y += dict[p][1]
			mov_x = mov_x/len(predictions[i])
			mov_y = mov_y / len(predictions[i])
			corners[i, 0] += mov_x
			corners[i, 1] += mov_y
	return corners

def createCalibrationDictionary():
	movs = dict()
	m = 0
	for delta_x in X_N:
		for delta_y in Y_N:
			movs[m] = [-delta_x, -delta_y]
			m += 1

	return movs

def visualize_corners(path_to_image, image_name):
	image = Image.open(PATH + path_to_image)
	corners = np.load(PATH + 'Results/corners_net3_' + image_name + '.npy')
	windows = getWindows(corners, image, 1)
	for window in windows:
		plt.imshow(window)
		plt.show()

def dispWindows(image, corners, disp):
	if (disp):
		fig, ax = plt.subplots(1)

		# Display the image
		ax.imshow(image)

		for corner in corners:
			# Create a Rectangle patch
			rect = Rectangle((corner[0], corner[1]), ORIGINAL_WINDOW_DIM, ORIGINAL_WINDOW_DIM, linewidth=1, edgecolor='g', facecolor='none')

			# Add the patch to the Axes
			ax.add_patch(rect)

		plt.show()

"""Testing"""
if __name__ == "__main__":
	image = Image.open("cornersDisplay/590.jpg")
	file_names = os.listdir("cornersDisplay/")
	for corners_name in file_names:
		if corners_name.endswith('.npy'):
			print(corners_name)
			corner = np.load('cornersDisplay/' + corners_name)
			dispWindows(image, corner, True)
	# 
	# print(file_names[0])
	# size_x,size_y = image.size
	# print(size_x, size_y)


	#visualize_corners("Data/TrainDotted/41.jpg", "41.jpg")

	# file_names = os.listdir("Data/Train/")
	# image = Image.open("Data/Train/" + file_names[0])
	# print(file_names[0])
	# size_x,size_y = image.size
	# print(size_x, size_y)
	# windows, corners = sliding_window_net_1(image)
	# print(windows.shape)
	# save_to_disk(windows, corners, 'Datasets/sliding_window_' + file_names[0] +'.h5')
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

	# movs = dict()
	# m = 0
	# for delta_x in X_N:
	# 	for delta_y in Y_N:
	# 		movs[m] = [-delta_x, -delta_y]
	# 		m += 1

	# predictions = [[0,8],[0,8],[0,8],[4],[3],[2],[5],[6],[3],[2]]
	# corners = np.zeros([10,2])
	# corners2 = calibrate(corners, predictions, movs)