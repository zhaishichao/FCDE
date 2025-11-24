import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def data_loader(file_path):


    df = pd.read_csv(file_path, comment='@', header=None)
    # 假设 df 是你的 DataFrame，最后一列是标签
    # 分离特征和标签
    pd.set_option('future.no_silent_downcasting', True)
    X = df.iloc[:, :-1].values                   # ndarray 格式
    y = df.iloc[:, -1].str.strip().replace({'negative': 0, 'positive': 1}).values
    return X, y

def data_preprocess(X, y, standard=False, normalize=False, random_state=42):

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    # 数据的标准化
    if standard:
        scaler_standard = StandardScaler()
        X_train = scaler_standard.fit_transform(X_train)
        X_test = scaler_standard.transform(X_test)

    # 数据的归一化
    if normalize:
        scaler_normalize = MinMaxScaler()  # 默认范围（0，1）
        X_train = scaler_normalize.fit_transform(X_train)
        X_test = scaler_normalize.transform(X_test)
    return X_train, X_test, y_train, y_test
