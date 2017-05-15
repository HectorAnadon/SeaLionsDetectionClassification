import numpy as np
from usefulFunctions import changeResolution, getWindow
import os
from PIL import Image
import matplotlib.pyplot as plt

def sliding_window_net_1(image, padding=10, window_size=100):
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

	return np.stack(windows), np.stack(corners)



"""Testing"""
if __name__ == "__main__":
	file_names = os.listdir("Data/Train/")
	image = Image.open("Data/Train/" + file_names[0])
	print(file_names[0])
	size_x,size_y = image.size
	print(size_x, size_y)
	windows, corners = sliding_window_net_1(image)
	print(windows.shape)
	print(corners[0])
	plt.imshow(windows[0])
	plt.show()
	plt.imshow(getWindow("Data/Train/" + file_names[0], 0, 0, 3))
	plt.show()
	print(corners[79542])
	plt.imshow(windows[79542])
	plt.show()
	print(corners[159083])
	plt.imshow(windows[159083])
	plt.show()
