from sklearn.tree import DecisionTreeClassifier

from metric import fit_pred, metric
from sklearn import clone
import pandas as pd
import os
from data_preprocess import data_loader, data_preprocess

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from config import datasetnames, num_run, file_path
from config import columns_dataset, columns_datasets, scoring

import warnings

warnings.filterwarnings("ignore")  # 忽略警告

# 保存路径
save_path = '../results/resample/dt/'
save_path_raw = save_path + 'raw/'
save_path_ros = save_path + 'ros/'
save_path_smote = save_path + 'smote/'
save_path_adasyn = save_path + 'adasyn/'
save_path_borderline_1 = save_path + 'borderline_1/'
save_path_borderline_2 = save_path + 'borderline_2/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_raw):
    os.makedirs(save_path_raw)
if not os.path.exists(save_path_ros):
    os.makedirs(save_path_ros)
if not os.path.exists(save_path_smote):
    os.makedirs(save_path_smote)
if not os.path.exists(save_path_adasyn):
    os.makedirs(save_path_adasyn)
if not os.path.exists(save_path_borderline_1):
    os.makedirs(save_path_borderline_1)
if not os.path.exists(save_path_borderline_2):
    os.makedirs(save_path_borderline_2)

if __name__ == '__main__':

    df_mean_raw = pd.DataFrame(columns=columns_datasets)
    df_mean_ros = pd.DataFrame(columns=columns_datasets)
    df_mean_smote = pd.DataFrame(columns=columns_datasets)
    df_mean_adasyn = pd.DataFrame(columns=columns_datasets)
    df_mean_borderline_1 = pd.DataFrame(columns=columns_datasets)
    df_mean_borderline_2 = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames):
        df_raw = pd.DataFrame(columns=columns_dataset)
        df_dg = pd.DataFrame(columns=columns_dataset)
        df_ds = pd.DataFrame(columns=columns_dataset)
        df_ros = pd.DataFrame(columns=columns_dataset)
        df_smote = pd.DataFrame(columns=columns_dataset)
        df_adasyn = pd.DataFrame(columns=columns_dataset)
        df_borderline_1 = pd.DataFrame(columns=columns_dataset)
        df_borderline_2 = pd.DataFrame(columns=columns_dataset)

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

            # ROS
            ros = RandomOverSampler(random_state=42 + i)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
            y_pred, y_prob = fit_pred(X_resampled, y_resampled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_ros = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_ros.loc[i] = [result_ros['f1_macro'], result_ros['g_mean'], result_ros['roc_auc_ovr']]

            # SMOTE
            smote = SMOTE(random_state=42 + i)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            y_pred, y_prob = fit_pred(X_resampled, y_resampled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_smote = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_smote.loc[i] = [result_smote['f1_macro'], result_smote['g_mean'], result_smote['roc_auc_ovr']]

            # ADASYN
            adasyn = ADASYN(random_state=42 + i)
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            y_pred, y_prob = fit_pred(X_resampled, y_resampled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_adasyn = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_adasyn.loc[i] = [result_adasyn['f1_macro'], result_adasyn['g_mean'], result_adasyn['roc_auc_ovr']]

            # Borderline-SMOTE-1
            borderline_smote = BorderlineSMOTE(random_state=42 + i)
            X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train)
            y_pred, y_prob = fit_pred(X_resampled, y_resampled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_borderline_1 = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_borderline_1.loc[i] = [result_borderline_1['f1_macro'], result_borderline_1['g_mean'],
                                      result_borderline_1['roc_auc_ovr']]
            # Borderline-SMOTE-2
            borderline_smote_2 = BorderlineSMOTE(kind='borderline-2', random_state=42 + i)
            X_resampled, y_resampled = borderline_smote_2.fit_resample(X_train, y_train)
            y_pred, y_prob = fit_pred(X_resampled, y_resampled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_borderline_2 = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_borderline_2.loc[i] = [result_borderline_2['f1_macro'], result_borderline_2['g_mean'],
                                      result_borderline_2['roc_auc_ovr']]

        df_mean_raw.loc[index] = [datasetname, num_instances, num_features, df_raw['F-measure'].mean(),
                                  df_raw['G-mean'].mean(), df_raw['AUC'].mean()]
        df_mean_ros.loc[index] = [datasetname, num_instances, num_features, df_ros['F-measure'].mean(),
                                  df_ros['G-mean'].mean(), df_ros['AUC'].mean()]
        df_mean_smote.loc[index] = [datasetname, num_instances, num_features, df_smote['F-measure'].mean(),
                                     df_smote['G-mean'].mean(), df_smote['AUC'].mean()]
        df_mean_adasyn.loc[index] = [datasetname, num_instances, num_features, df_adasyn['F-measure'].mean(),
                                     df_adasyn['G-mean'].mean(), df_adasyn['AUC'].mean()]
        df_mean_borderline_1.loc[index] = [datasetname, num_instances, num_features, df_borderline_1['F-measure'].mean(),
                                           df_borderline_1['G-mean'].mean(), df_borderline_1['AUC'].mean()]
        df_mean_borderline_2.loc[index] = [datasetname, num_instances, num_features, df_borderline_2['F-measure'].mean(),
                                           df_borderline_2['G-mean'].mean(), df_borderline_2['AUC'].mean()]
        # 保存结果到csv文件
        df_raw.to_csv(save_path_raw + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_ros.to_csv(save_path_ros + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_smote.to_csv(save_path_smote + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_adasyn.to_csv(save_path_adasyn + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_borderline_1.to_csv(save_path_borderline_1 + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_borderline_2.to_csv(save_path_borderline_2 + datasetname + '.csv', encoding='utf_8_sig', index=False)
        # 每处理完一个数据集，保存平均结果
        df_mean_raw.to_csv(save_path + 'mean_raw.csv', encoding='utf_8_sig', index=False)
        df_mean_ros.to_csv(save_path + 'mean_ros.csv', encoding='utf_8_sig', index=False)
        df_mean_smote.to_csv(save_path + 'mean_smote.csv', encoding='utf_8_sig', index=False)
        df_mean_adasyn.to_csv(save_path + 'mean_adasyn.csv', encoding='utf_8_sig', index=False)
        df_mean_borderline_1.to_csv(save_path + 'mean_borderline_1.csv', encoding='utf_8_sig', index=False)
        df_mean_borderline_2.to_csv(save_path + 'mean_borderline_2.csv', encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
