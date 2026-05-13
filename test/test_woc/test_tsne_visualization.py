import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import os
from data_preprocess import data_loader, data_preprocess
from smote_variants import DGSMOTE
from smote_variants.borderline_smote import BorderSMOTE
from smote_variants.gp_smote_woc import DSSMOTE
import warnings

from config import num_run, evol_parameter, file_path, datasetnames_2
from smote_variants.gp_smote_c4_v2.visualize import tsne_visualization_binary

warnings.filterwarnings("ignore")  # 忽略警告

columns_dataset = ['time']
columns_datasets = ['数据集', 'time']

# 保存路径
save_path = '../results/gp/tsne/'

if __name__ == '__main__':

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames_2):

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # GPSMOTE
            dg = DGSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            X_train_syn_ds, y_train_syn_ds = dg.fit_resample_synthesis_only()
            y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
                                      F'{datasetname}' + '_dg_' + str(i + 1))
            # 将y_train_res_ds拼接到X_train_res_ds最后一列
            X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_border.csv',
                                                index=False)

            # # GPSMOTE
            # border = BorderSMOTE(k_neighbors=5, random_state=42 + i)
            # X_train_syn_ds, y_train_syn_ds = border.fit_resample_only_synthetic(X_train, y_train)
            # y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            # X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            # y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            # tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
            #                           F'{datasetname}' + '_border_' + str(i + 1))
            # # 将y_train_res_ds拼接到X_train_res_ds最后一列
            # X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            # pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_border.csv',
            #                                     index=False)
            #
            # # GPSMOTE
            # ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter, remove_gi=1)
            # X_train_syn_ds, y_train_syn_ds = ds.fit_resample_synthesis_only()
            # y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            # X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            # y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            # tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
            #                           F'{datasetname}' + '_re_g1_' + str(i + 1))
            # # 将y_train_res_ds拼接到X_train_res_ds最后一列
            # X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            # pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_re_g1.csv',
            #                                     index=False)
            # # GPSMOTE
            # ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter, remove_gi=2)
            # X_train_syn_ds, y_train_syn_ds = ds.fit_resample_synthesis_only()
            # y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            # X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            # y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            # tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
            #                           F'{datasetname}' + '_re_g2_' + str(i + 1))
            # # 将y_train_res_ds拼接到X_train_res_ds最后一列
            # X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            # pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_re_g2.csv',
            #                                     index=False)
            # # GPSMOTE
            # ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter, remove_gi=3)
            # X_train_syn_ds, y_train_syn_ds = ds.fit_resample_synthesis_only()
            # y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            # X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            # y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            # tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
            #                           F'{datasetname}' + '_re_g3_' + str(i + 1))
            # # 将y_train_res_ds拼接到X_train_res_ds最后一列
            # X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            # pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_re_g3.csv',
            #                                     index=False)
            # # GPSMOTE
            # ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter, remove_gi=4)
            # X_train_syn_ds, y_train_syn_ds = ds.fit_resample_synthesis_only()
            # y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            # X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            # y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            # tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
            #                           F'{datasetname}' + '_re_g4_' + str(i + 1))
            # # 将y_train_res_ds拼接到X_train_res_ds最后一列
            # X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            # pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_re_g4.csv',
            #                                     index=False)
            #
            # # GPSMOTE
            # ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter, remove_gi=0)
            # X_train_syn_ds, y_train_syn_ds = ds.fit_resample_synthesis_only()
            # y_train_syn_ds = [2 for _ in range(len(y_train_syn_ds))]
            # X_train_res_ds = np.vstack((X_train, X_train_syn_ds))
            # y_train_res_ds = np.hstack((y_train, y_train_syn_ds))
            # tsne_visualization_binary(X_train_res_ds, y_train_res_ds, save_path + datasetname,
            #                           F'{datasetname}' + '_all_' + str(i + 1))
            # # 将y_train_res_ds拼接到X_train_res_ds最后一列
            # X_train_res_ds = np.hstack((X_train_res_ds, np.array([y_train_res_ds]).T))
            # # 保存X_train_res_ds和y_train_res_ds原始数据为csv
            # pd.DataFrame(X_train_res_ds).to_csv(save_path + datasetname + '_' + str(i + 1) + '_X_train_res_all.csv',
            #                                     index=False)

    print('########\t 结束执行！\t########')
