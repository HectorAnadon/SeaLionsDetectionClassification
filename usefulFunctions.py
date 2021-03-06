
from PIL import Image
import matplotlib.pyplot as plt
import os
import pdb
import cv2
import skimage.feature
import pandas as pd
import time
from global_variables import *
import numpy as np


#RETURNS THE IMAGE SIZE IN EACH RESOLUTION
def sizeInfo(image,resolutions):

    # image = ORIGINAL IMAGE WITH FULL RESOLUTION
    # resolutions = LIST WITH QUALITIES USED IN RANGE (0,1]

    #ORIGINAL SIZE
    sizes = []
    size_x_original,size_y_original = image.size
    sizes.append((size_x_original,size_y_original))

    for i in range(1,len(resolutions)):
        size_x = int(round(size_x_original * resolutions[i]))
        size_y = int(round(size_y_original * resolutions[i]))
        resized = image.resize((size_x, size_y))
        size_x, size_y = resized.size
        sizes.append((size_x, size_y))

    return sizes

def getImageSize(resolution_lvl):
    #DEFINE RESOLUTION
    if resolution_lvl == 1:
        quality = 1
    elif resolution_lvl == 2:
        quality = 0.5
    else:
        quality = 0.25 
    return int(round(ORIGINAL_WINDOW_DIM * quality))

def get_resolution_level(net):
    if net == 1:
        resolution_lvl = 3
    elif net == 2:
        resolution_lvl = 2
    else:
        resolution_lvl = 1
    return resolution_lvl

def changeResolution(image, resolution_lvl):
    # Get new window size
    #print(image.size)
    new_size = getImageSize(resolution_lvl)
    # Resize window
    image = image.resize((new_size, new_size)) 
    return image

def getLabel(image_name, coordinates, x0, y0, s):
    label = [0,1]
    classes = CLASSES
    for lion_class in classes:
        for lion in range(len(coordinates[lion_class][image_name])):
            if coordinates[lion_class][image_name][lion][0] > x0 and coordinates[lion_class][image_name][lion][0] < (x0 + s)\
            and coordinates[lion_class][image_name][lion][1] > y0 and coordinates[lion_class][image_name][lion][1] < (y0 + s):
                label = [1,0]
    return label

#1.CROP OUT AND MODIFY RESOLUTION OF AN IMAGE
def cropAndChangeResolution(image,image_name,x0,y0,s,resolution_lvl):
    return changeResolution(image.crop((x0,y0,x0+s,y0+s)), resolution_lvl)


#EXTRACT SEA LION COORDINATES
def extractCoordinates(path, image_name, defaultParent="Data/"):

    #DATAFRAME TO STORE THE RESULTS
    classes = CLASSES
    coordinates_df = pd.DataFrame(index=list([image_name]), columns=classes)

    # READ TRAIN AND DOTTED IMAGES
    image_1 = cv2.imread(path + defaultParent + "TrainDotted/" + image_name)
    image_2 = cv2.imread(path + defaultParent + "Train/" + image_name)

    # COMPUTE ABSOLUTE DIFFERENCE BETWEEN TRAIN AND TRAIN DOTTED
    image_3 = cv2.absdiff(image_1,image_2)

    # MASK OUT BACKENED REGIONS FROM TRAIN DOTTED
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)

    # CONVERT TO GRAYSCALE TO BE ACCEPTED BY SKIMAGE
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    # DETECT BLOBS
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

    adult_males = []
    subadult_males = []
    pups = []
    juveniles = []
    adult_females = []

    for blob in blobs:
        # GET THE COORDINATES OF EACH BLOB
        y, x, s = blob

        # GET COLOR OF THE PIXEL IN THE CENTER OF THE BLOB
        g, b, r = image_1[int(y)][int(x)][:]

        # DETECT THE COLOR
        if r > 200 and g < 50 and b < 50:  # RED = ADULT MALES
            adult_males.append((int(x), int(y)))
        elif r > 200 and g > 200 and b < 50:  # MAGENTA = SUBADULT MALES
            subadult_males.append((int(x), int(y)))
        elif r < 100 and g < 100 and 150 < b < 200:  # GREEN = PUPS
            pups.append((int(x), int(y)))
        elif r < 100 and 100 < g and b < 100:  # BLUE = JUVENILES
            juveniles.append((int(x), int(y)))
        elif r < 150 and g < 50 and b < 100:  # BROWN = FEMALES
            adult_females.append((int(x), int(y)))

    coordinates_df["adult_males"][image_name] = adult_males
    coordinates_df["subadult_males"][image_name] = subadult_males
    coordinates_df["adult_females"][image_name] = adult_females
    coordinates_df["juveniles"][image_name] = juveniles
    coordinates_df["pups"][image_name] = pups

    return  coordinates_df

def plot_loss_functions():
    path = PATH + 'Results/'
    file_names = os.listdir(path)
    legend = []
    plt.gca().set_color_cycle(['red', 'red', 'green', 'green', 'blue', 'blue'])
    for loss_np in file_names:
        #if loss_np.startswith('loss') and loss_np.endswith('.npy'):
        if 'loss_calib' in loss_np and loss_np.endswith('.npy'):
            legend.append(loss_np) #TODO: process the name
            logs = np.load(path + loss_np)
            loss = []
            val_loss = []
            for log in logs:
                loss.append(log['loss'])
                val_loss.append(log['val_loss'])
            plt.plot(loss)
            plt.plot(val_loss, '--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss function - Calibration nets')            
    plt.legend(['Train net 1', 'Validation net 1', 'Train net 2', 'Validation net 2', 'Train net 3', 'Validation net 3'])
    plt.show()
    



#####################
#      EXAMPLE      #
#####################

if __name__ == '__main__':
    plot_loss_functions()

    # file_names = os.listdir(PATH + "Data/Train/")
    # image = Image.open(PATH + "Data/Train/" + file_names[1])

    # a = sizeInfo(image,resolutions = [1,0.5,0.25])
    # print(a)

    # image = cropAndChangeResolution("Data/",file_names[1],0,0,3328,1)
    # coordinates = extractCoordinates("Data/", file_names[1])
    # label = getLabel(file_names[1], coordinates, 0, 0, 1000);
    # print(label)
    # plt.imshow(image)
    # plt.show()
