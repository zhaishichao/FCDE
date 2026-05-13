from sklearn.neighbors import KNeighborsClassifier

import time
import pandas as pd
import os
from smote_variants.dg_smote import DGSMOTE
from data_preprocess import data_loader, data_preprocess
from smote_variants.gp_smote_population import DSSMOTE
import warnings

from config import datasetnames, num_run, evol_parameter, file_path

warnings.filterwarnings("ignore")  # 忽略警告

columns_dataset = ['time']
columns_datasets = ['数据集', 'time']

# 保存路径
save_path = '../results/gp/runtime/'
save_path_dg = save_path + 'dg/'
save_path_ds = save_path + 'ds/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_dg):
    os.makedirs(save_path_dg)
if not os.path.exists(save_path_ds):
    os.makedirs(save_path_ds)

if __name__ == '__main__':

    df_mean_dg = pd.DataFrame(columns=columns_datasets)
    df_mean_ds = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames):
        df_dg = pd.DataFrame(columns=columns_dataset)
        df_ds = pd.DataFrame(columns=columns_dataset)

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # DGSMOTE
            dg = DGSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            start = time.perf_counter()
            _, _ = dg.fit_resample()
            end = time.perf_counter()
            df_dg.loc[i] = [end - start]

            # GPSMOTE
            ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            start = time.perf_counter()
            _, _ = ds.fit_resample()
            end = time.perf_counter()
            df_ds.loc[i] = [end - start]

        df_mean_dg.loc[index] = [datasetname, df_dg['time'].mean()]
        df_mean_ds.loc[index] = [datasetname, df_ds['time'].mean()]

        # 保存结果到csv文件
        df_dg.to_csv(save_path_dg + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_ds.to_csv(save_path_ds + datasetname + '.csv', encoding='utf_8_sig', index=False)
        # 每处理完一个数据集，保存平均结果
        df_mean_dg.to_csv(save_path + 'mean_dg.csv', encoding='utf_8_sig', index=False)
        df_mean_ds.to_csv(save_path + 'mean_ds.csv', encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
