import cv2
import pandas as pd
import AnalysisFunction as AF
import os
import datetime as dt
import glob

# set up directory and search through all folders within
parent_dir = r"C:\Users\wzhuo\Dropbox (Personal)\Research\Projects\BCScreening\Data\TissueClassification\sample_mutant_candidate"

worm_num = 1
pop_ID = 0

for pop_dir in [[f.path, f.name] for f in os.scandir(parent_dir) if f.is_dir()]:

    pop_ID += 1
    strain_name = pop_dir[1]
    currentPath = pop_dir[0]

    try:
        os.chdir(currentPath)
        print("Processing images in directory " + currentPath)
    except:
        print("Could not access directory " + currentPath)
        continue

    # Go through all .png files in the given directory

    cols1 = ['pop_ID', 'worm_num', 'Cut_off', 'est_len', 'int_ratio', 'head_first', 'bk_mean', 'body_mean',
             'body_max_width','body_std', 'body_perct95', 'body_median', 'body_sum', 'aspect_ratio']
    worm_data = pd.dataframe(columns=cols1)

    cols2 = ['worm_num','sec_1','sec_2','sec_3','sec_4','sec_5','sec_6','sec_7','sec_8','sec_9','sec_10',]
    worm_sec_data = pd.dataframe(columns=cols2)

    for filename in glob.glob('*.png'):

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        [worm_data, worm_sec_data] = AF.worm_analysis(img, worm_data, worm_sec_data)
        worm_num += 1



