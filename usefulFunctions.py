
from PIL import Image
import matplotlib.pyplot as plt
import os
import pdb
import cv2
import skimage.feature
import pandas as pd

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

#1.CROP OUT AND MODIFY RESOLUTION OF AN IMAGE + 2.RETURN LABEL - NO SEA LION [0,1] / SEA LION [1,0])
def changeResolution(path,image_name,x0,y0,h,v,resolution_lvl):

    # image_name
    # x0 = TOP LEFT CORNER IN X AXIS
    # y0 = TOP LEFT CORNER IN Y AXIS
    # h = NUMBER OF PIXEL IN X AXIS OF THE CROPPED IMAGE
    # v = NUMBER OF PIXEL IN Y AXIS OF THE CROPPED IMAGE
    # resolution_lvl = SPECIFY RESOLUTION AMONG ALL POSSIBILITIES

    image = Image.open(path +"Train/"+ image_name)

    #DEFINE RESOLUTION
    if resolution_lvl == 1:
        quality = 1
    elif resolution_lvl == 2:
        quality = 0.5
    else:
        quality = 0.25

    #CHECK IF THERE IS A SEA LION IN THE IMAGE
    coordinates = extractCoordinates(path,image_name)
    label = [0,1]
    classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
    for lion_class in classes:
        for lion in range(len(coordinates[lion_class][image_name])):
            if coordinates[lion_class][image_name][lion][0] > x0 and coordinates[lion_class][image_name][lion][0] < (x0 + h)\
                and coordinates[lion_class][image_name][lion][1] > y0 and coordinates[lion_class][image_name][lion][1] < (y0 + v):
                label = [1,0]

    #CROP OUT
    image = image.crop((x0,y0,x0+h,y0+v))

    #CHANGE RESOLUTION
    size_x,size_y = image.size
    size_x = int(round(size_x*quality))
    size_y = int(round(size_y*quality))

    #RESIZE
    image = image.resize((size_x, size_y))

    return  image,label

#EXTRACT SEA LION COORDINATES
def extractCoordinates(path, image_name):

    #DATAFRAME TO STORE THE RESULTS
    classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
    coordinates_df = pd.DataFrame(index=list([image_name]), columns=classes)

    # READ TRAIN AND DOTTED IMAGES
    image_1 = cv2.imread(path +"TrainDotted/" + image_name)
    image_2 = cv2.imread(path + "Train/" + image_name)

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

#####################
#      EXAMPLE      #
#####################

os.chdir("/Users/albertbou/SeaLionsDetectionClassification/Data/Train/")
file_names = os.listdir("/Users/albertbou/SeaLionsDetectionClassification/Data/Train/")
image = Image.open(file_names[0])

a = sizeInfo(image,resolutions = [1,0.5,0.25])
print(a)

image,label = changeResolution("/Users/albertbou/SeaLionsDetectionClassification/Data/",file_names[0],2250,2250,100,100,3)
print(label)
plt.imshow(image)
plt.show()
