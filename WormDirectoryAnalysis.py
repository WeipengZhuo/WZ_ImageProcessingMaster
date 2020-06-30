# Branched off of Faye Clever's LiveBodyAnalysisV1 Master on 6/25/20

# test image: D:\03192020_AllRawData\ClonalImagingData\02_01_2020_2nd_round_clonal_imaging\Group2\Trial2\Block1\kah160\Mut3_2_Snapshot55

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import scipy.ndimage as sp
import AnalysisFunction as AF
import glob
import os
import sqlite3
from datetime import datetime

# Set up cmd arguments for the user
parser = argparse.ArgumentParser(description = 'Enter absolute path and background coordinates')
parser.add_argument('path', type = str, help = 'Absolute folder path of the head folder (ex. /mnt/c/Users/fhcle/Documents/GeorgiaTech/McGrath_Lab/Image_Annotation/Python')
parser.add_argument('bkCoords', type = int, nargs = '+', help = 'Four ints representing coordinates of the image that are part of the background: (left_col, top_row, right_col, bottom_row)')
args = parser.parse_args()
bkCoords = args.bkCoords

# Set up csv and headers for analysis
o = open('out.csv','w')
o.write('Folder,Name,Num,threshold,num_regions,cut_off,est_len,bkMeanInt,wormMeanInt,seg2int,seg9int,tipWidth1,tipWidth2,tipWidthRatio,avgStdevInt,')
o.write('BodMeanInt,BodAvgWidth,BodStdInt,BodPerct95,BodMedianInt,BKMeanInt,BKAvgWidth,BKStdInt,BKPerct95,BKMedianInt\n')

# Iterate through all folders (not files) in the given cmd head node directory.
# Variable abs_directory is an absolute path.
for abs_directory in [f.path for f in os.scandir(args.path) if f.is_dir()]:
    folder = abs_directory.split('/')[-1]

    # If there is a pycache folder in your directory (like mine), you want to skip it
    if abs_directory.endswith('__pycache__'):
        continue

    # Enter each directory
    try:
        os.chdir(abs_directory)
        print("Processing images in directory " + abs_directory)
    except:
        print("Could not access directory " + abs_directory)
    continue

img_num = 1
for filename in glob.glob('*.png'):

    if img_num % 10 == 0:
        print("\tProcessing image #" + str(img_num))
        img_num += 1

        #####################################################################################
        # Step 1: read image
        #####################################################################################

        # Process a single image
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # If image cannot be processed, continue to the next image in the folder
        if img is None:
            print("Could not read the image " + filename)
            continue

        # Write file name to output file only if its within the first 100 files
        name = filename.split('.')[0]
        num = ''.join([i for i in name.split('_')[-1] if i.isdigit()])
        # For the purpose of feature exploration, we only want the files that we
        # manually analyzed and which are in the ground truth data
        if int(num) > 100:
            continue
        o.write(folder + ',')
        o.write(name + ',')
        o.write(num + ',')

o.close()
