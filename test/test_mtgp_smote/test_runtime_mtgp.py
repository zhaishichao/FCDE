from sklearn.neighbors import KNeighborsClassifier

import time
import pandas as pd
import os
from smote_variants.dg_smote import DGSMOTE
from data_preprocess import data_loader, data_preprocess
from smote_variants.gp_smote_population import DSSMOTE
import warnings

from config import datasetnames, num_run, evol_parameter, file_path
from smote_variants.mtgp_smote.mtgp_smote import MTGPSMOTESampler

warnings.filterwarnings("ignore")  # 忽略警告

columns_dataset = ['time']
columns_datasets = ['数据集', 'time']

# 保存路径
save_path = '../results/mtgp/runtime/'
save_path_mtgp = save_path + 'mtgp/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_mtgp):
    os.makedirs(save_path_mtgp)
if not os.path.exists(save_path_mtgp):
    os.makedirs(save_path_mtgp)

if __name__ == '__main__':

    df_mean_mtgp = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames):
        df_mtgp = pd.DataFrame(columns=columns_dataset)

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # GPSMOTE
            mtgp = MTGPSMOTESampler(
                pop_size=512,  # 论文 512，此处缩小以加快演示
                n_generations=100,  # 论文 100
                cx_rate=0.7,
                mut_rate=0.3,
                tournament_k=3,
                max_depth=4,
                random_state=42 + i)
            start = time.perf_counter()
            _, _ = mtgp.fit_resample(X_train, y_train)
            end = time.perf_counter()
            df_mtgp.loc[i] = [end - start]

        df_mean_mtgp.loc[index] = [datasetname, df_mtgp['time'].mean()]

        # 保存结果到csv文件
        df_mtgp.to_csv(save_path_mtgp + datasetname + '.csv', encoding='utf_8_sig', index=False)
        # 每处理完一个数据集，保存平均结果
        df_mean_mtgp.to_csv(save_path + 'mean_mtgp.csv', encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
