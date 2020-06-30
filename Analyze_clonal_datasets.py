import cv2
import pandas as pd
import AnalysisFunction as AF
import os
import datetime as dt
import glob
import matplotlib as plt

######################################
# analyzing clonal population part 1 #
######################################
parent_dir = r"D:\AllRawData_03192020\ClonalImagingData\BC_clonalImaging_7_30_18"

#Defining data frames
cols1 = ['pop_ID', 'worm_num', 'cut_off', 'est_len', 'int_ratio', 'head_first', 'head_perct95', 'bk_mean', 'body_mean',
         'body_max_width', 'body_std', 'body_perct95', 'body_median', 'body_sum', 'aspect_ratio']
cols2 = ['sec_1', 'sec_2', 'sec_3', 'sec_4', 'sec_5', 'sec_6', 'sec_7', 'sec_8', 'sec_9', 'sec_10',
         'sec_11', 'sec_12', 'sec_13', 'sec_14', 'sec_15', 'sec_16', 'sec_17', 'sec_18', 'sec_19', 'sec_20', ]
cols3 = ['pop_ID','strain_name','exp_ID','experiment_date', 'group', 'block']
worm_data = pd.DataFrame(columns=cols1)
worm_sec_data = pd.DataFrame(columns=cols2)
population_data = pd.DataFrame(columns=cols3)

#Initializing counts
worm_num = 0
pop_ID = 0
exp_ID = 0

#Open directory for all part 1 images to find experimental folders
for exp_dir in [[f.path, f.name] for f in os.scandir(parent_dir) if f.is_dir()]:
    exp_ID += 1
    exp_date = exp_dir[1]
    os.chdir(exp_dir[0])
    print("Processing images in directory " + exp_dir[0])

    # Open directory for each experimental folder to find population folders
    for pop_dir in [[f.path, f.name] for f in os.scandir(exp_dir[0]) if f.is_dir()]:
        pop_ID += 1
        strain_name = pop_dir[1]
        currentPath = pop_dir[0]
        try:
            os.chdir(currentPath)
            print("Processing images in directory " + currentPath)
        except:
            print("Could not access directory " + pop_dir[1])
            continue
        population_data = population_data.append(pd.Series([pop_ID, strain_name, exp_ID, exp_date, str(0), str(0)], index=cols3),
                                                 ignore_index=True)

        # Go through all .png files in the given population
        for filename in glob.glob('*.png'):
            if filename == 'Capture.png':
                break
            try:
                worm_num += 1
                [worm_data, worm_sec_data] = AF.worm_analysis(pop_ID, worm_num, filename, worm_data, worm_sec_data)
            except:
                print("could not process image: " + filename)
                continue

######################################
# analyzing clonal population part 2 #
######################################
parent_dir = r"D:\AllRawData_03192020\ClonalImagingData\clonal_imaging_02_01_2020_2nd_round"

#Defining data frames
cols1 = ['pop_ID', 'worm_num', 'cut_off', 'est_len', 'int_ratio', 'head_first', 'head_perct95', 'bk_mean', 'body_mean',
         'body_max_width', 'body_std', 'body_perct95', 'body_median', 'body_sum', 'aspect_ratio']
cols2 = ['sec_1', 'sec_2', 'sec_3', 'sec_4', 'sec_5', 'sec_6', 'sec_7', 'sec_8', 'sec_9', 'sec_10',
         'sec_11', 'sec_12', 'sec_13', 'sec_14', 'sec_15', 'sec_16', 'sec_17', 'sec_18', 'sec_19', 'sec_20', ]
cols3 = ['pop_ID','strain_name','exp_ID','experiment_date', 'group', 'block']
worm_data2 = pd.DataFrame(columns=cols1)
worm_sec_data2 = pd.DataFrame(columns=cols2)
population_data2 = pd.DataFrame(columns=cols3)

#Initializing counts
worm_num = 0
pop_ID = 0
exp_ID = 0

#Open directory for all part 1 set to find group folders
for grp_dir in [[f.path, f.name] for f in os.scandir(parent_dir) if f.is_dir()]:
    group_num = grp_dir[1]
    print("Processing: " + grp_dir[0])
    for Trial_dir in [[f.path, f.name] for f in os.scandir(grp_dir[0]) if f.is_dir()]:
        for Block_dir in [[f.path, f.name] for f in os.scandir(Trial_dir[0]) if f.is_dir()]:
            exp_ID += 1
            os.chdir(Block_dir[0])
            for pop_dir in [[f.path, f.name] for f in os.scandir(Block_dir[0]) if f.is_dir()]:
                pop_ID += 1
                strain_name = pop_dir[1]
                currentPath = pop_dir[0]
                try:
                    os.chdir(currentPath)
                    print("Processing images in directory " + currentPath)
                except:
                    print("Could not access directory " + pop_dir[1])
                    continue
                population_data2 = population_data2.append(pd.Series([pop_ID, strain_name, exp_ID, exp_date, group_num, Block_dir[1]], index=cols3),
                                                         ignore_index=True)

                # Go through all .png files in the given population
                for filename in glob.glob('*.png'):
                    if filename == 'Capture.PNG':
                        break
                    try:
                        worm_num += 1
                        [worm_data2, worm_sec_data2] = AF.worm_analysis(pop_ID, worm_num, filename, worm_data2, worm_sec_data2)
                    except:
                        print("could not process image: " + filename)
                        continue

writer = pd.ExcelWriter(r'C:\Users\wzhuo\PycharmProjects\WZ_ImageProcessingMaster\output.xlsx')
worm_data.to_excel(writer)