from data_preprocess import data_loader, data_preprocess
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

file_path = '../datasets/dat/'
save_path = './tsne_results_1128_p_a/'
# datasetnames = ['iris0', 'ecoli1', 'glass0', 'glass1', 'haberman', 'pima', 'segment0', 'vowel0', 'wisconsin', 'yeast1']
datasetnames = ['ecoli1']
if __name__ == '__main__':
    for datasetname in datasetnames:
        X, y = data_loader(file_path + datasetname + '.dat')
        with open('../log.txt', 'a') as f:
            print(f'数据集：{datasetname}', file=f)
            print(f'实例数量：{X.shape[0]}', file=f)
            print(f'特征数量：{X.shape[1]}', file=f)

        X_train, X_test, y_train, y_test = data_preprocess(X, y,random_state=42)

        from sklearn.preprocessing import StandardScaler
        from visualize import tsne_visualization_binary

        scaler = StandardScaler()

        # 使用t-SNE进行降维
        X_tsne = tsne_visualization_binary(scaler.fit_transform(X_train), y_train,
                                           save_path=save_path + datasetname,
                                           filename=datasetname, perplexity=30)  # 传入的是标准化后的特征数据

        from config import EvolutionaryParameterConfig

        from de import DSSMOTE_P_A

        evol_parameter = EvolutionaryParameterConfig(30, 0.8, 0.2, 100, False)

        dgpa = DSSMOTE_P_A(X=X_train, y=y_train, evol_parameter=evol_parameter)
        X_syn, y_syn = dgpa.fit_resample_synthesis_only()

        print(X_syn)
        print(y_syn)

        # 可视化
        y_syn = [2 for _ in range(len(y_syn))]
        X_train_resampled = np.vstack((X_train, X_syn))
        y_train_resampled = np.hstack((y_train, y_syn))
        # 4. 使用t-SNE进行降维
        X_tsne_resampled_p_a = tsne_visualization_binary(scaler.fit_transform(X_train_resampled), y_train_resampled,
                                                         save_path=save_path + datasetname,
                                                         filename=datasetname + '_dgpa', perplexity=30)
