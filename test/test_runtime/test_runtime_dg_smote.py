import time
import os
import warnings
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from data_preprocess import data_loader, data_preprocess
from smote_variants.dg_smote import DGSMOTE
from config import datasetnames_final_1, EvolutionaryParameterConfig

warnings.filterwarnings("ignore")

num_run = 10
POPSIZE = 30
CXPB = 0.8
MUTPB = 0.2
NGEN = 100
evol_parameter = EvolutionaryParameterConfig(POPSIZE, CXPB, MUTPB, NGEN, verbose=False)

columns_dataset = ['time']
columns_datasets = ['数据集', 'time']

save_path = '../test/results/runtime/dg_smote/'
os.makedirs(save_path, exist_ok=True)

if __name__ == '__main__':
    df_mean = pd.DataFrame(columns=columns_datasets)
    print('########\t DGSMOTE 运行时间测试开始！\t########')

    for index, datasetname in enumerate(datasetnames_final_1):
        df_time = pd.DataFrame(columns=columns_dataset)
        print(f'##########\t 正在处理：{datasetname} \t##########')

        X, y = data_loader('datasets/dat/' + datasetname + '.dat')
        for i in range(num_run):
            X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42 + i)

            dg = DGSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
            start = time.perf_counter()
            _, _ = dg.fit_resample()
            end = time.perf_counter()
            df_time.loc[i] = [end - start]

        df_mean.loc[index] = [datasetname, df_time['time'].mean()]
        df_time.to_csv(save_path + datasetname + '.csv', encoding='utf_8_sig', index=False)
        df_mean.to_csv(save_path + 'mean.csv', encoding='utf_8_sig', index=False)

    print('########\t DGSMOTE 运行时间测试结束！\t########')
