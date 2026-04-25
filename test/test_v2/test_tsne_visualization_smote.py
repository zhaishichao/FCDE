import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import os

from smote.blind_smote_res_only import BlindSMOTE
from data_preprocess import data_loader, data_preprocess
import warnings

from config import datasetnames_1, num_run, file_path
from smote.gp_smote_c4_v2.visualize import tsne_visualization_binary
from smote.mtgp_smote_res_only.mtgp_smote import MTGPSMOTESampler

warnings.filterwarnings("ignore")  # 忽略警告

# 保存路径
save_path = '../results/tsne/evol/'
save_path_bs = save_path + 'bs/'
save_path_mt = save_path + 'mt/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_bs):
    os.makedirs(save_path_bs)
if not os.path.exists(save_path_mt):
    os.makedirs(save_path_mt)

if __name__ == '__main__':

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames_1):

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # BlindSMOTE
            bs = BlindSMOTE(
                pop_size=30,
                n_gen=100, verbose=True,
                classifier=KNeighborsClassifier(),
                random_state=42 + i)
            _, _, X_train_syn_bs = bs.fit_resample(X_train, y_train)
            y_train_syn_bs = [2 for _ in range(len(X_train_syn_bs))]
            X_train_res_bs = np.vstack((X_train, X_train_syn_bs))
            y_train_res_bs = np.hstack((y_train, y_train_syn_bs))
            tsne_visualization_binary(X_train_res_bs, y_train_res_bs, save_path + datasetname,
                                      F'{datasetname}' + '_bs_' + str(i + 1))
            # 将y_train_res_ds拼接到X_train_res_ds最后一列
            X_train_res_bs = np.hstack((X_train_res_bs, np.array([y_train_res_bs]).T))
            # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            pd.DataFrame(X_train_res_bs).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_bs.csv',
                                                index=False)

            # BlindSMOTE
            mtgp = MTGPSMOTESampler(
                pop_size=30,  # 论文 512，此处缩小以加快演示
                n_generations=100,  # 论文 100
                cx_rate=0.7,
                mut_rate=0.3,
                tournament_k=3,
                max_depth=4,
                random_state=42 + i)
            X_train_syn_mt = mtgp.fit_resample(X_train, y_train)
            y_train_syn_mt = [2 for _ in range(len(X_train_syn_mt))]
            X_train_res_mt = np.vstack((X_train, X_train_syn_mt))
            y_train_res_mt = np.hstack((y_train, y_train_syn_mt))
            tsne_visualization_binary(X_train_res_mt, y_train_res_mt, save_path + datasetname,
                                      F'{datasetname}' + '_mt_' + str(i + 1))
            # 将y_train_res_ds拼接到X_train_res_ds最后一列
            X_train_res_mt = np.hstack((X_train_res_mt, np.array([y_train_res_mt]).T))
            # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            pd.DataFrame(X_train_res_mt).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_mt.csv',
                                                index=False)


    print('########\t 结束执行！\t########')
