# IMPORT PACKAGES
import numpy as np
import pdb
from global_variables import *

def non_max_suppression_slow(corners,overlapThresh,confidenceScores): #COMMON VALUES FOR "overlapThresh" ARE BETWEEN 0.3 AND 0.5

    # IF THERE ARE NO BOXES, RETURN EMPY LIST
    if corners.shape[0] == 0:
        return []

    # INITIALIZE THE LIST OF PICKED INDEXES
    pick = []

    # GRAB THE COORDINATES OF THE BOUNDING BOXES
    x1 = corners[:, 0]
    y1 = corners[:, 1]
    x2 = corners[:, 0] + ORIGINAL_WINDOW_DIM
    y2 = corners[:, 1] + ORIGINAL_WINDOW_DIM

    # COMPUTE THE AREA OF THE BOUNDING BOXES AND SORT THE BOUNDING
    # BOXES BY THE BOTTON-RIGHT Y-COORDINATE OF THE BOUNDING BOX
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidenceScores)
    #idxs = np.argsort(y2)

    # KEEP LOOPING WHILE SOME IDEXES STILL REMAIN IN THE LIST
    while len(idxs) > 0:
        # GRAB THE LAST INDEX IN THE LIST, ADD THE INDEX
        # VALUE TO THE LIST OF PICKED INDEXES, THEN INITIALIZE
        # THE SUPPRESSION LIST (INDEXES THAT WILL BE DELETED)
        # USING THE LAST INDEX
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # LOOP OVER ALL INDEXES IN THE LIST
        for pos in range(0, last):
            # GRAB THE CURRENT INDEX
            j = idxs[pos]

            # FIND THE LARGEST (x,y) COORDINATES FOR THE START OF
            # THE BOUNDING BOX AND THE SMALLEST (x,y) COORDINATES
            # FOR THE END OF THE BOUNDING BOX
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # COMPUTE THE WIDTH AND THE HEIGHT OF THE BOUNDING BOX
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # COMPUTE THE RATIO OF OVERLAP BETWEEN THE COMPUTED
            # BOUNDING BOX AND THE BOUNDING BOX IN THE AREA LIST
            overlap = float(w * h) / area[j]

            # IF THERE IS SUFFICIENT OVERLAP, SUPPRESS THE
            # CURRENT BOUNDING BOX
            if overlap > overlapThresh:
                suppress.append(pos)

        # DELETE ALL INDEXES IN THE LIST THAT ARE IN THE
        # SUPPRESSION LIST
        idxs = np.delete(idxs, suppress)
        # RETURN ONLY THE BOUNDING BOXES THAT WERE PICKED
    return corners[pick,:]

def non_max_suppression_fast(corners,overlapThresh,confidenceScores): #COMMON VALUES FOR "overlapThresh" ARE BETWEEN 0.3 AND 0.5
    # if there are no corners, return an empty list
    if len(corners) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if corners.dtype.kind == "i":
        corners = corners.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = corners[:, 0]
    y1 = corners[:, 1]
    x2 = corners[:, 0] + ORIGINAL_WINDOW_DIM
    y2 = corners[:, 1] + ORIGINAL_WINDOW_DIM

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidenceScores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return corners[pick,:]


###################
#       TEST      #
###################
if __name__ == '__main__':
    images = np.random.randn(30,100,100,3)
    scores = np.random.rand(30)
    corner = np.random.randn(10,2)+100
    corners = np.append(corner,np.random.randn(10,2),axis=0)
    corners = np.append(corners,np.random.randn(10,2)+500,axis=0)

    picked_corners = non_max_suppression_fast(corners,0.3,scores)
    for i in range(len(scores)):
        print(str(i+1) +':'+str(scores[i]))
    print('#'*50)
    for i in range(len(scores)):
        print(str(i+1) +': '+str(corners[i]))
    print('#'*50)
    print(picked_corners)