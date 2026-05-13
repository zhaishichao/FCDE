import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn import clone
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from metric import fit_pred, metric
from data_preprocess import data_loader, data_preprocess
from config import file_path, columns_dataset, columns_datasets, scoring
from smote_variants.blind_smote import BlindSMOTE
from smote_variants.mtgp_smote.mtgp_smote import MTGPSMOTESampler

warnings.filterwarnings("ignore")


def create_clf(i, clf_type):
    """根据类型字符串创建分类器。

    Parameters
    ----------
    i : int, 迭代序号，用于设置 random_state
    clf_type : {'knn', 'dt', 'svm'}
    """
    if clf_type == 'knn':
        return KNeighborsClassifier()
    elif clf_type == 'dt':
        return DecisionTreeClassifier(random_state=42 + i)
    elif clf_type == 'svm':
        return SVC(kernel='linear', C=1.0, probability=True, random_state=42 + i)
    raise ValueError(f"Unknown classifier type: {clf_type}")


def create_sampler(i, sampler_type, clf_type=None):
    """根据类型字符串创建SMOTE采样器。

    Parameters
    ----------
    i : int, 迭代序号
    sampler_type : {'bs', 'mtgp'}
        bs: BlindSMOTE, mtgp: MTGPSMOTESampler
    clf_type : str or None
        分类器类型，仅 BlindSMOTE 需要（其内部使用分类器评估适应度）。
    """
    if sampler_type == 'bs':
        return BlindSMOTE(
            pop_size=30, n_gen=30,
            classifier=create_clf(i, clf_type),
            random_state=42 + i)
    elif sampler_type == 'mtgp':
        return MTGPSMOTESampler(
            pop_size=30, n_generations=100,
            cx_rate=0.7, mut_rate=0.3, tournament_k=3, max_depth=4,
            random_state=42 + i)
    raise ValueError(f"Unknown sampler type: {sampler_type}")


def run_experiment(clf_type, sampler_type, dataset_names, n_runs):
    """运行SMOTE过采样实验，遍历数据集并评估分类性能。

    save_path、save_subfolder、mean_filename 由 clf_type 和 sampler_type
    自动推导，无需手动指定。

    Parameters
    ----------
    clf_type : {'knn', 'dt', 'svm'}, 分类器类型
    sampler_type : {'bs', 'mtgp'}, 采样器类型
    dataset_names : list, 数据集名称列表
    n_runs : int, 重复运行次数
    """
    save_path = f'../results/{sampler_type}/{clf_type}/'
    mean_filename = f'mean_{sampler_type}.csv'
    os.makedirs(save_path, exist_ok=True)

    df_mean = pd.DataFrame(columns=columns_datasets)

    print('########\t 开始执行！\t########')

    for index, datasetname in enumerate(dataset_names):
        df_run = pd.DataFrame(columns=columns_dataset)

        print('##########\t', '正在处理：', datasetname, '\t##########')
        X, y = data_loader(file_path + datasetname + '.dat')
        num_instances, num_features = X.shape

        for i in range(n_runs):
            clf = create_clf(i, clf_type)
            X_train, X_test, y_train, y_test = data_preprocess(
                X, y, standard=True, random_state=42 + i)

            sampler = create_sampler(i, sampler_type, clf_type)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            X_shuffled, y_shuffled = shuffle(X_res, y_res, random_state=42 + i)

            y_pred, y_prob = fit_pred(
                X_shuffled, y_shuffled.astype('int'),
                X_test=X_test, clf=clone(clf), soft_lable=True)
            result = metric(y_test.astype('int'), y_pred, y_prob, scoring)

            df_run.loc[i] = [result['f1_macro'], result['g_mean'], result['roc_auc_ovr']]

        df_mean.loc[index] = [
            datasetname, num_instances, num_features,
            df_run['F-measure'].mean(),
            df_run['G-mean'].mean(),
            df_run['AUC'].mean()
        ]

        df_run.to_csv(os.path.join(save_path, datasetname + '.csv'),
                       encoding='utf_8_sig', index=False)
        df_mean.to_csv(os.path.join(save_path, mean_filename),
                        encoding='utf_8_sig', index=False)

    print('########\t 结束执行！\t########')
