"""
Title: Rooftops detection in Google Maps Image
Author: Jitender Singh Virk (Virksaab)
Date created: 26 July, 2018
Last Modified: 26 July, 2018
"""
import numpy as np
import cv2
import os
import time


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

# GET ALL IMAGES PATH FROM FOLDERS
imagepaths = []
for dirpath, dirnames, filenames in os.walk('Satellite Images of different areas in delhi'):
    for filename in filenames:
        # print(filename)
        imagepaths.append(os.path.join(dirpath, filename))
        
# ITERATE OVER ALL IMAGES
for i, imgpath in enumerate(imagepaths):
    # GET IMAGE AND RESIZE
    bgrimg = cv2.imread(imgpath)
    bgrimg = cv2.resize(bgrimg, (800, 600), interpolation=cv2.INTER_CUBIC)
    gray = cv2.imread(imgpath, 0)
    gray = cv2.resize(gray, (800, 600), interpolation=cv2.INTER_CUBIC)

    start = time.time()

    # SHARPEN
    kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
    sgray = cv2.filter2D(gray, -1, kernel_sharp)
    sbgrimg = cv2.filter2D(bgrimg, -1, kernel_sharp)


    # THRESHOLDING
    ret, mask = cv2.threshold(sgray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # EDGES
    edges = auto_canny(mask)
    invedges = cv2.bitwise_not(edges)


    # REFINE MASK
    mieg = cv2.bitwise_and(mask, invedges)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.bitwise_and(mieg, opening)
    refined = cv2.bitwise_not(refined)

    # CONVERT MASK TO MATCH WITH ORIGNAL IMAGE DIMENSIONS
    refined3d = sbgrimg.copy()
    vidx, hidx = refined.nonzero()
    for ii in range(len(vidx)):
        refined3d[vidx[ii]][hidx[ii]][0] = 0
        refined3d[vidx[ii]][hidx[ii]][1] = 0
        refined3d[vidx[ii]][hidx[ii]][2] = 0

    print("time taken by image {1}: {0:.4f} sec".format(time.time() - start, i))

    # DISPLAY RESULTS
    # stacked2d = np.hstack((sgray, refined))
    stacked3d = np.hstack((sbgrimg, refined3d))
    # cv2.imshow("SHARPEN", stacked3d)

    # WRITE IMAGES TO DISK
    cv2.imwrite('rooftops_mask/{}.jpg'.format(i), stacked3d)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()