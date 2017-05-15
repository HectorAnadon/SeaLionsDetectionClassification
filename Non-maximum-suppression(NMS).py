

# IMPORT PACKAGES
import numpy as np
import pdb

def non_max_suppression_slow(corners,overlapThresh,size): #COMMON VALUES FOR "overlapThresh" ARE BETWEEN 0.3 AND 0.5

    # IF THERE ARE NO BOXES, RETURN EMPY LIST
    if len(images) == 0:
        return []

    # INITIALIZE THE LIST OF PICKED INDEXES
    pick = []

    # GRAB THE COORDINATES OF THE BOUNDING BOXES
    x1 = corners[:, 0]
    y1 = corners[:, 1]
    x2 = corners[:, 0] + size
    y2 = corners[:, 1] + size

    # COMPUTE THE AREA OF THE BOUNDING BOXES AND SORT THE BOUNDING
    # BOXES BY THE BOTTON-RIGHT Y-COORDINATE OF THE BOUNDING BOX
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

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
        for pos in xrange(0, last):
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


###################
#       TEST      #
###################
if __name__ == '__main__':
    images = np.random.randn(20,100,100,3)
    corners = np.random.randn(20,2)
    picked_corners = non_max_suppression_slow(corners,0.3,100)
    print(picked_corners)