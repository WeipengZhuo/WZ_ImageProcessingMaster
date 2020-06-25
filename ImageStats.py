import numpy as np
import sys
import sklearn.decomposition as skl
import sklearn.preprocessing as sklp
import matplotlib.pyplot as plt
import pandas as pd


def stats(img):
    MeanInt = np.mean(img[np.nonzero(img)])
    stdInt = np.std(img[np.nonzero(img)])
    perct95 = np.percentile(img[np.nonzero(img)], 95)
    medianInt = np.percentile(img[np.nonzero(img)], 50)

    [row, col] = np.nonzero(img)
    avgWidth = max(row) - min(row)

    SumInt = np.sum(img[np.nonzero(img)])

    return [MeanInt, avgWidth, stdInt, perct95, medianInt, SumInt]


def standardize(df, features):
    scaler = sklp.StandardScaler()
    scaled_dt = pd.DataFrame(scaler.fit_transform(df.values))
    scaled_dt.columns = features
    return scaled_dt


def my_correlation(original_dt, pca_data, main_feature):
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


def pca_plot(dt, features, target, name_tag=''):
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