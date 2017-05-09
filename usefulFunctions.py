
from PIL import Image
import matplotlib.pyplot as plt
import os
import pdb

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

#CROP OUT AND MODIFY RESOLUTION OF AN IMAGE
def changeResolution(image,x0,y0,h,v,resolution_lvl):

    # image
    # x0 = TOP LEFT CORNER IN X AXIS
    # y0 = TOP LEFT CORNER IN Y AXIS
    # h = NUMBER OF PIXEL IN X AXIS OF THE CROPPED IMAGE (AFTER SESIZE!)
    # v = NUMBER OF PIXEL IN Y AXIS OF THE CROPPED IMAGE (AFTER SESIZE!)
    # resolution_lvl = SPECIFY RESOLUTION AMONG ALL POSSIBILITIES (maybe that should be an input parameter...)

    #DEFINE RESOLUTION
    if resolution_lvl == 1:
        quality = 1
    elif resolution_lvl == 2:
        quality = 0.5
    else:
        quality = 0.1

    #CROP OUT
    image = image.crop((x0,y0,x0+h,y0+v))

    #CHANGE RESOLUTION
    size_x,size_y = image.size
    size_x = int(round(size_x*quality))
    size_y = int(round(size_y*quality))

    #RESIZE
    image = image.resize((size_x, size_y))


    return  image

#####################
#      EXAMPLE      #
#####################

os.chdir("/Users/albertbou/SeaLionsDetectionClassification/Data/Train/")
file_names = os.listdir("/Users/albertbou/SeaLionsDetectionClassification/Data/Train/")
image = Image.open(file_names[0])

a = sizeInfo(image,resolutions = [1,0.5,0.1])
print(a)

image = changeResolution(image,2000,2000,1000,1000,3)
plt.imshow(image)
plt.show()