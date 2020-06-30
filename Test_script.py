import cv2
import pandas as pd
import AnalysisFunction as AF
import os
import datetime as dt
import glob
import matplotlib as plt


# set up directory and search through all folders within
parent_dir = r"C:\Users\wzhuo\Dropbox (Personal)\Research\Projects\BCScreening\Data\TissueClassification\sample_mutant_candidate"

worm_num = 1
pop_ID = 0

cols1 = ['pop_ID', 'worm_num', 'cut_off', 'est_len', 'int_ratio', 'head_first', 'head_perct95', 'bk_mean', 'body_mean',
         'body_max_width', 'body_std', 'body_perct95', 'body_median', 'body_sum', 'aspect_ratio']
worm_data = pd.DataFrame(columns=cols1)

cols2 = ['sec_1', 'sec_2', 'sec_3', 'sec_4', 'sec_5', 'sec_6', 'sec_7', 'sec_8', 'sec_9', 'sec_10',
         'sec_11', 'sec_12', 'sec_13', 'sec_14', 'sec_15', 'sec_16', 'sec_17', 'sec_18', 'sec_19', 'sec_20', ]
worm_sec_data = pd.DataFrame(columns=cols2)

for pop_dir in [[f.path, f.name] for f in os.scandir(parent_dir) if f.is_dir()]:

    pop_ID += 1
    strain_name = pop_dir[1]
    currentPath = pop_dir[0]



    try:
        os.chdir(currentPath)
        print("Processing images in directory " + currentPath)
    except:
        print("Could not access directory " + pop_dir[1])
        continue

    # Go through all .png files in the given directory

    for filename in glob.glob('*.png'):

        # print("Processing image: " + filename)

        # img = cv2.imread(filename, -1)
        try:
            [worm_data, worm_sec_data] = AF.worm_analysis(pop_ID, worm_num, filename, worm_data, worm_sec_data)
            worm_num += 1
        except:
            print("couldnt process image: " + filename)
            continue



