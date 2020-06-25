# Go through all .png files in the given directory

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import scipy.ndimage as sp
import AnalysisFunctions as AF
import glob
import os
import sqlite3
from datetime import datetime


def worm_analysis(img):
    # Select background region (safe region is rows 0-100, columns 0-2000)
    bkCoords = [448, 449, 881, 37]
    bk = img[bkCoords[1]:(bkCoords[1] + bkCoords[3]), bkCoords[0]:(bkCoords[0] + bkCoords[2])]

    #####################################################################################
    # Step 2: threshold image
    #####################################################################################

    # Otsu's thresholding to get the image to be binary black and white.
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Binary thresholding with an adjusted Otsu's threshold
    ret1, th1 = cv2.threshold(img, ret * 0.65, 255, cv2.THRESH_BINARY)

    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(2,1,2)
    plt.imshow(th1)
    plt.title(str(ret1))
    plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
    plt.show()

    # Take Otsu's thresholded image, th2, and do dilating/morphological opening.
    kernel_2 = np.ones((2, 2), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)
    ImDia = cv2.dilate(th1, kernel_2, iterations=1)
    ImOpen = cv2.morphologyEx(ImDia, cv2.MORPH_OPEN, kernel_5)
    filled = sp.morphology.binary_fill_holes(ImOpen)

    optional: plot the results
    plt.subplot(4,1,1)
    plt.imshow(th1)
    plt.title('Thresholded Image')
    plt.subplot(4,1,2)
    plt.imshow(ImDia)
    plt.title('Dilated Image (kernel size = 2)')
    plt.subplot(4,1,3)
    plt.imshow(ImOpen)
    plt.title('Image after Morphological Opening (kernel size = 5)')
    plt.subplot(4,1,4)
    plt.imshow(filled)
    plt.title('filled')
    plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
    plt.show()

    #####################################################################################
    # Step 3: isolate worm
    #####################################################################################

    labeled_array, num_regions = sp.label(filled)
    max_size = 0
    max_region = 0

    # Iterate through the regions identified by sp.label, excluding 0, the background
    for x in range(1, num_regions + 1):
        # Get the size of each region (count) and update the maximums
        count = np.count_nonzero(labeled_array == x)
        if (count > max_size):
            max_size = count
            max_region = x

    # Isolate the desired region by dividing each pixel by the number of the largest
    # region. If a pixel belongs to the largest region, this value will be one.
    # Generate a matrix of True and False, and turn it into 0s and 1s using .astype(int)
    justWorm = (np.divide(labeled_array, max_region) == 1).astype(int)

    # Multiply your binary mask by the original image to get just the worm body
    image_gray = np.multiply(justWorm, img)

    Plot resulting image
    worm = np.divide(image_gray, 255)
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(2,1,2)
    plt.imshow(worm)
    plt.title('Worm Body Only')
    plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
    plt.show()

    #####################################################################################
    # Step 4: Feature Extraction
    #####################################################################################

    # Is the image cut off? Is there a nonzero number in any of the outer rows/columns?
    Cutoff = 1
    row1 = image_gray[0]
    row2 = image_gray[image_gray.shape[0] - 1]
    for row in image_gray:
        if row[0] >= 1:
            Cutoff = 3
        if row[image_gray.shape[1] - 1] >= 1:
            Cutoff = 3
    for position in row1:
        if position >= 1:
            Cutoff = 3
    for position in row2:
        if position >= 1:
            Cutoff = 3

    # Is the animal head or tail first?
    Sec2Image = image_gray[min(row):max(row), (min(col) + (2) * est_len // 10):(min(col) + (3) * est_len // 10)]
    Sec9Image = image_gray[min(row):max(row), (min(col) + (9) * est_len // 10):(min(col) + (10) * est_len // 10)]
    [MeanInt, avgWidth, stdInt, Sec_2_perct95, medianInt] = AF.stats(Sec2Image)
    [MeanInt, avgWidth, stdInt, Sec_9_perct95, medianInt] = AF.stats(Sec9Image)
    int_ratio = Sec_9_perct95 / Sec_2_perct95

    # Is the animal coiled?
    [row, col] = np.nonzero(image_gray)
    est_len = max(col) - min(col)
    # aspect_ratio of animal
    aspect_ratio = est_len / np.max(Sec20[1][0], Sec20[1][-1])

    # FeatureExtraction

    # background intensity
    bk_mean = np.mean(bk)

    # Full worm body stats
    [BodMeanInt, BodMaxidth, BodStdInt, BodPerct95, BodMedianInt] = AF.stats(
        image_gray[min(row):max(row), min(col):max(col)])

    # Sectional Analysis
    Sec20 = [[], [], [], [], []]
    for i in range(20):
        # divide the worm body into 20 equal segments according to estimated length, est_len
        SecImage = image_gray[min(row):max(row), (min(col) + (i) * est_len // 20):(min(col) + (i + 1) * est_len // 20)]
        [MeanInt, avgWidth, stdInt, perct95, medianInt] = AF.stats(SecImage)
        Sec20[0].append(MeanInt)
        Sec20[1].append(avgWidth)
        Sec20[2].append(stdInt)
        Sec20[3].append(perct95)
        Sec20[4].append(medianInt)
