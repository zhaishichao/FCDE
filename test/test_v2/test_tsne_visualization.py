import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import os
from smote_variants.dg_smote import DGSMOTE
from data_preprocess import data_loader, data_preprocess
from smote_variants.gp_smote_population import DSSMOTE
import warnings

from config import datasetnames, num_run, evol_parameter, file_path
from smote_variants.gp_smote_population.visualize import tsne_visualization_binary

warnings.filterwarnings("ignore")  # 忽略警告

columns_dataset = ['time']
columns_datasets = ['数据集', 'time']

# 保存路径
save_path = '../results/gp/tsne/'
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

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames):

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # DGSMOTE
            dg = DGSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            X_train_syn_dg, y_train_syn_dg = dg.fit_resample_synthesis_only()
            y_train_syn_dg = [2 for _ in range(len(y_train_syn_dg))]
            X_train_res_dg = np.vstack((X_train, X_train_syn_dg))
            y_train_res_dg = np.hstack((y_train, y_train_syn_dg))
            tsne_visualization_binary(X_train_res_dg, y_train_res_dg, save_path + datasetname, F'{datasetname}' + '_dg_' + str(i+1))
            # 将y_train_res_dg拼接到X_train_res_dg最后一列
            X_train_res_dg = np.hstack((X_train_res_dg, np.array([y_train_res_dg]).T))
            # 保存X_train_res_dg和y_train_res_dg原始数据为csv
            pd.DataFrame(X_train_res_dg).to_csv(save_path + datasetname + '_' + str(i+1) + '_X_train_res_dg.csv', index=False)

            # GPSMOTE
            ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            X_train_syn_ds, y_train_syn_ds = ds.fit_resample_synthesis_only()
            y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname, F'{datasetname}' + '_ds_' + str(i+1))
            # 将y_train_res_ds拼接到X_train_res_ds最后一列
            X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i+1) + '_X_train_res_ds.csv', index=False)

    print('########\t 结束执行！\t########')
