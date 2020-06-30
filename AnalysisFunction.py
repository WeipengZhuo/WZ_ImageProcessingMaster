def worm_analysis(pop_ID, worm_num, filename, worm_data, worm_sec_data):

    import pandas as pd
    import numpy as np
    import cv2
    import scipy.ndimage as sp

    grayscale = cv2.imread(filename, 0)
    bkCoords = [448, 449, 881, 37]
    bk = original[bkCoords[1]:(bkCoords[1] + bkCoords[3]), bkCoords[0]:(bkCoords[0] + bkCoords[2])]

    # Otsu's thresholding to get the image to be binary black and white.
    ret, th = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Binary thresholding with an adjusted Otsu's threshold
    ret1, th1 = cv2.threshold(grayscale, ret * 0.65, 255, cv2.THRESH_BINARY)


    # Take Otsu's thresholded image, th2, and do dilating/morphological opening.
    kernel_2 = np.ones((2, 2), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)
    ImDia = cv2.dilate(th1, kernel_2, iterations=1)
    ImOpen = cv2.morphologyEx(ImDia, cv2.MORPH_OPEN, kernel_5)
    filled = sp.morphology.binary_fill_holes(ImOpen)

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


    justWorm = (np.divide(labeled_array, max_region) == 1).astype(int)

    original = cv2.imread(filename, -1)
    image_gray = np.multiply(justWorm, original)

    #####################################################################################
    # Step 4: Feature Extraction
    #####################################################################################

    # Is the image cut off? Is there a nonzero number in any of the outer rows/columns?
    cut_off = 0
    row1 = image_gray[0]
    row2 = image_gray[image_gray.shape[0] - 1]
    for row in image_gray:
        if row[0] >= 1:
            cut_off = 1
        if row[image_gray.shape[1] - 1] >= 1:
            cut_off = 1
    for position in row1:
        if position >= 1:
            cut_off = 1
    for position in row2:
        if position >= 1:
            cut_off = 1

    # worm length
    [row, col] = np.nonzero(image_gray)

    min_row = min(row)
    max_row = max(row)
    min_col = min(col)
    max_col = max(col)

    est_len = max_col - min_col

    # head or tail metric
    Sec2Image = image_gray[min_row:max_row, (min_col + (2) * est_len // 10):(min_col + (3) * est_len // 10)]
    Sec9Image = image_gray[min_row:max_row, (min_col + (9) * est_len // 10):(min_col + (10) * est_len // 10)]

    Sec_2_perct95 = np.percentile(Sec2Image[np.nonzero(Sec2Image)], 95)
    Sec_9_perct95 = np.percentile(Sec9Image[np.nonzero(Sec9Image)], 95)
    int_ratio = Sec_9_perct95 / Sec_2_perct95

    # flip animals tail first
    if int_ratio >= 1.3:
        head_first = 1
    else:
        head_first = 0
        image_gray = np.fliplr(image_gray)

    [row, col] = np.nonzero(image_gray)
    min_row = min(row)
    max_row = max(row)
    min_col = min(col)
    max_col = max(col)

    head_image = image_gray[min_row:max_row, (min_col + 4 * est_len // 5):(min_col + 5 * est_len // 5)]
    head_perct95 = np.percentile(head_image[np.nonzero(head_image)], 95)

    # FeatureExtraction

    # background intensity
    bk_mean = np.mean(bk)

    # Full worm body stats
    [body_mean, body_max_width, body_std, body_perct95, body_median, body_sum] = stats(
        image_gray[min_row:max_row, min_col:max_col])

    # Sectional Analysis
    avgWidth = []
    for i in range(20):
        # divide the worm body into 20 equal segments according to estimated length, est_len
        SecImage = image_gray[min_row:max_row, (min_col + (i) * est_len // 20):(min_col + (i + 1) * est_len // 20)]

        MeanInt = np.mean(SecImage[np.nonzero(SecImage)])

        [row2, col2] = np.nonzero(SecImage)
        avgWidth.append(max(row2) - min(row2))


    # aspect_ratio of animal
    aspect_ratio = est_len / np.max([avgWidth[0], avgWidth[-1]])

    row1 = [pop_ID, worm_num, cut_off, est_len, int_ratio, head_first, head_perct95, bk_mean, body_mean,
            body_max_width, body_std, body_perct95, body_median, body_sum, aspect_ratio]
    worm_data = worm_data.append(pd.Series(row1, index=worm_data.columns), ignore_index=True)


    worm_sec_data = worm_sec_data.append(pd.Series(MeanInt, index=worm_sec_data.columns), ignore_index=True)

    return worm_data, worm_sec_data


########################################################################################################################

def stats(img):
    import numpy as np

    MeanInt = np.mean(img[np.nonzero(img)])
    stdInt = np.std(img[np.nonzero(img)])
    perct95 = np.percentile(img[np.nonzero(img)], 95)
    medianInt = np.percentile(img[np.nonzero(img)], 50)
    [row, col] = np.nonzero(img)
    avgWidth = max(row) - min(row)
    SumInt = np.sum(img[np.nonzero(img)])
    return [MeanInt, avgWidth, stdInt, perct95, medianInt, SumInt]


########################################################################################################################

def standardize(df, features):
    import pandas as pd
    import sklearn.preprocessing as sklp

    scaler = sklp.StandardScaler()
    scaled_dt = pd.DataFrame(scaler.fit_transform(df.values))
    scaled_dt.columns = features
    return scaled_dt


########################################################################################################################

def my_correlation(original_dt, pca_data, main_feature):
    import pandas as pd
    import matplotlib.pyplot as plt

    corr = pd.concat([original_dt, pca_data], axis=1).corr()
    plt.matshow(corr, fignum=10)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.rcParams.update({'font.size': 10})
    plt.gcf().set_size_inches(14.5, 14.5)
    plt.title('Correlation of PCs with original Features: Analyzing ' + main_feature, pad=60)
    plt.savefig('PCA_correlation_figures/pca_corr_' + main_feature + '.png')
    plt.close()
    return corr


########################################################################################################################

def pca_plot(dt, features, target, name_tag=''):
    import pandas as pd
    import sklearn.decomposition as skl
    import matplotlib.pyplot as plt

    x = dt.loc[:, features]
    y = dt.loc[:, target]
    scaled_dt = standardize(x, features)
    pca = skl.PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_dt)
    print(pca.explained_variance_ratio_)
    # Create data from PCA output
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])
    # principalDf.to_excel(writer2, sheet_name = 'PCA2_raw_df_' + target)
    finalDf = pd.concat([principalDf, dt[target]], axis=1)
    # finalDf.to_excel(writer2, sheet_name = 'PCA2_final_df_' + target)
    # Graph PCA
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA: ' + target[0], fontsize=20)
    colors = ['r', 'g']
    for i, color in zip((0, 1), colors):
        # For the selected color, you only want one strain to be plotted
        indicesToKeep = finalDf[target[0]] == i
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
                   c=color, s=50)
    ax.legend(['Not ' + target[0], target[0]])
    ax.grid()
    fig.savefig('PCA_correlation_figures/PCA2_' + target[0] + name_tag + '.png')
    fig.show()

    return fig, pca, my_correlation(dt[features], principalDf, target[0] + name_tag)
