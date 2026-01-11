from sklearn.tree import DecisionTreeClassifier

from metric import fit_pred, metric
from sklearn.utils import shuffle
import pandas as pd
import os
from smote.dg_smote import DGSMOTE
from data_preprocess import data_loader, data_preprocess
from smote.gp_smote_c4_v2 import DSSMOTE
import warnings
from sklearn import clone

from config import datasetnames, num_run, evol_parameter, file_path
from config import columns_dataset, columns_datasets, scoring

warnings.filterwarnings("ignore")  # 忽略警告

# 保存路径
save_path = '../results/v2/dt/'
save_path_raw = save_path + 'raw/'
save_path_dg = save_path + 'dg/'
save_path_ds = save_path + 'ds/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_raw):
    os.makedirs(save_path_raw)
if not os.path.exists(save_path_dg):
    os.makedirs(save_path_dg)
if not os.path.exists(save_path_ds):
    os.makedirs(save_path_ds)

if __name__ == '__main__':

    df_mean_raw = pd.DataFrame(columns=columns_datasets)
    df_mean_dg = pd.DataFrame(columns=columns_datasets)
    df_mean_ds = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames):
        df_raw = pd.DataFrame(columns=columns_dataset)
        df_dg = pd.DataFrame(columns=columns_dataset)
        df_ds = pd.DataFrame(columns=columns_dataset)

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = DecisionTreeClassifier(random_state=42 + i)
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # RAW 原始数据
            y_pred, y_prob = fit_pred(X_train, y_train.astype('int'), X_test=X_test, clf=clone(clf), soft_lable=True)
            result_raw = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_raw.loc[i] = [result_raw['f1_macro'], result_raw['g_mean'], result_raw['roc_auc_ovr']]
            # DGSMOTE
            dg = DGSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            X_train_resampled, y_train_resampled = dg.fit_resample()
            X_shuffled, y_shuffled = shuffle(X_train_resampled, y_train_resampled,
                                             random_state=42 + i)  # random_state 保证可复现性
            y_pred, y_prob = fit_pred(X_shuffled, y_shuffled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_dg = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_dg.loc[i] = [result_dg['f1_macro'], result_dg['g_mean'], result_dg['roc_auc_ovr']]
            # GPSMOTE
            ds = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            X_train_resampled, y_train_resampled = ds.fit_resample()
            X_shuffled, y_shuffled = shuffle(X_train_resampled, y_train_resampled,
                                             random_state=42 + i)  # random_state 保证可复现性
            y_pred, y_prob = fit_pred(X_shuffled, y_shuffled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_ds = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_ds.loc[i] = [result_ds['f1_macro'], result_ds['g_mean'], result_ds['roc_auc_ovr']]

        df_mean_raw.loc[index] = [datasetname, num_instances, num_features, df_raw['F-measure'].mean(),
                                  df_raw['G-mean'].mean(), df_raw['AUC'].mean()]
        df_mean_dg.loc[index] = [datasetname, num_instances, num_features, df_dg['F-measure'].mean(),
                                 df_dg['G-mean'].mean(), df_dg['AUC'].mean()]
        df_mean_ds.loc[index] = [datasetname, num_instances, num_features, df_ds['F-measure'].mean(),
                                 df_ds['G-mean'].mean(), df_ds['AUC'].mean()]
        # 保存结果到csv文件
        df_raw.to_csv(save_path_raw + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_dg.to_csv(save_path_dg + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_ds.to_csv(save_path_ds + datasetname + '.csv', encoding='utf_8_sig', index=False)
        # 每处理完一个数据集，保存平均结果
        df_mean_raw.to_csv(save_path + 'mean_raw.csv', encoding='utf_8_sig', index=False)
        df_mean_dg.to_csv(save_path + 'mean_dg.csv', encoding='utf_8_sig', index=False)
        df_mean_ds.to_csv(save_path + 'mean_ds.csv', encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
