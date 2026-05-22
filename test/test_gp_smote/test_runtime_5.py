from sklearn.neighbors import KNeighborsClassifier

import time
import pandas as pd
import os
from data_preprocess import data_loader, data_preprocess
from smote_variants.gp_smote_population import DSSMOTE
import warnings

from config import datasetnames, file_path, EvolutionaryParameterConfig

warnings.filterwarnings("ignore")  # 忽略警告
num_run = 5
POPSIZE = 500  # 种群大小
CXPB = 0.8  # 交叉概率
MUTPB = 0.2  # 变异概率
NGEN = 100  # 迭代次数
verbose = False  # 是否打印信息

evol_parameter = EvolutionaryParameterConfig(POPSIZE, CXPB, MUTPB, NGEN, verbose)
columns_dataset = ['time']
columns_datasets = ['数据集', 'time']

# 保存路径
save_path = '../results/gp_big_pop/runtime_5/'
save_path_ds = save_path + 'ds/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_ds):
    os.makedirs(save_path_ds)

if __name__ == '__main__':

    df_mean_ds = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames):
        df_ds = pd.DataFrame(columns=columns_dataset)

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # GPSMOTE
            ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            start = time.perf_counter()
            _, _ = ds.fit_resample()
            end = time.perf_counter()
            df_ds.loc[i] = [end - start]

        df_mean_ds.loc[index] = [datasetname, df_ds['time'].mean()]

        df_ds.to_csv(save_path_ds + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_mean_ds.to_csv(save_path + 'mean_ds.csv', encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
