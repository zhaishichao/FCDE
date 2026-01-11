from data_preprocess import data_loader, data_preprocess
from visualize import tsne_visualization_binary
from config import EvolutionaryParameterConfig
from gp_smote_test import DSSMOTE
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

file_path = '../datasets/dat/'
save_path = 'results/0105_5/'
datasetnames = ['iris0', 'ecoli1', 'glass0', 'glass1', 'haberman', 'pima', 'segment0', 'vowel0', 'wisconsin', 'yeast1',
                'shuttle-c0-vs-c4', 'australian', 'heart', 'phoneme', 'ring', 'spambase', 'wdbc']
if __name__ == '__main__':
    print('########\t 开始执行！\t########')
    for datasetname in datasetnames:
        X, y = data_loader(file_path + datasetname + '.dat')
        X_train, X_test, y_train, y_test = data_preprocess(X, y, standard=True, random_state=42)
        # 使用t-SNE进行降维
        X_tsne = tsne_visualization_binary(X_train, y_train, save_path + datasetname, datasetname)  # 传入的是标准化后的特征数据

        # GPSMOTE
        evol_parameter = EvolutionaryParameterConfig(30, 0.8, 0.2, 100, False)
        dss = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
        X_syn, y_syn = dss.fit_resample_synthesis_only()
        # 可视化, 将DSSMOTE生成的数据与原始数据进行拼接
        y_syn = [2 for _ in range(len(y_syn))]
        X_train_resampled = np.vstack((X_train, X_syn))
        y_train_resampled = np.hstack((y_train, y_syn))
        # 4. 使用t-SNE进行降维
        X_tsne_resampled_p_a = tsne_visualization_binary(X_train_resampled, y_train_resampled, save_path + datasetname,
                                                         datasetname + '_dss')
        dss.curve_fitting(save_path + datasetname + '/', 'convergence_curve', datasetname)
        with open(save_path +datasetname + '/'+datasetname +'.txt', 'w') as file:
            file.write(f'实例数量：{X.shape[0]}\n')
            file.write(f'特征数量：{X.shape[1]}\n')
    print('########\t 结束执行！\t########')
