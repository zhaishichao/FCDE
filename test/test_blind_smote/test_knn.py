from sklearn.neighbors import KNeighborsClassifier

from metric import fit_pred, metric
from sklearn.utils import shuffle
import pandas as pd
import os

from smote_variants.blind_smote import BlindSMOTE
from data_preprocess import data_loader, data_preprocess
import warnings
from sklearn import clone

from config import datasetnames_final_1, num_run, file_path
from config import columns_dataset, columns_datasets, scoring

warnings.filterwarnings("ignore")  # 忽略警告

# 保存路径
save_path = '../results/bs/knn/'
save_path_bs = save_path + 'bs/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_bs):
    os.makedirs(save_path_bs)

if __name__ == '__main__':

    df_mean_bs = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(datasetnames_final_1):
        df_bs = pd.DataFrame(columns=columns_dataset)

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape
        for i in range(num_run):
            clf = KNeighborsClassifier()
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            # BlindSMOTE
            bs = BlindSMOTE(
                pop_size=30,
                n_gen=100,verbose=True,
                classifier=KNeighborsClassifier(),
                random_state=42 + i)
            X_train_resampled, y_train_resampled = bs.fit_resample(X_train, y_train)
            X_shuffled, y_shuffled = shuffle(X_train_resampled, y_train_resampled, random_state=42 + i)
            y_pred, y_prob = fit_pred(X_shuffled, y_shuffled.astype('int'), X_test=X_test, clf=clone(clf),
                                      soft_lable=True)
            result_dg = metric(y_test.astype('int'), y_pred, y_prob, scoring)
            df_bs.loc[i] = [result_dg['f1_macro'], result_dg['g_mean'], result_dg['roc_auc_ovr']]

            df_mean_bs.loc[index] = [datasetname, num_instances, num_features, df_bs['F-measure'].mean(),
                                     df_bs['G-mean'].mean(), df_bs['AUC'].mean()]

            # 保存结果到csv文件
            df_bs.to_csv(save_path_bs + datasetname + '.csv', encoding='utf_8_sig', index=False)
            # 每处理完一个数据集，保存平均结果
            df_mean_bs.to_csv(save_path + 'mean_dg.csv', encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
