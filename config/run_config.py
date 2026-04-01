# 实验参数设置
from config import EvolutionaryParameterConfig

datasetnames_1 = ['appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass1', 'ring', 'car-good', 'yeast5',
                  'vowel0']

datasetnames_2 = ['glass0', 'new-thyroid1', 'shuttle-6-vs-2-3', 'heart', 'ecoli-0-1-4-6-vs-5', 'spambase', 'banana',
                  'phoneme']

datasetnames_3 = ['haberman', 'ecoli1', 'ecoli2', 'ecoli3', 'ionosphere', 'titanic', 'segment0', 'chess', 'flare-F']

# datasetnames_3 = ['kddcup-rootkit-imap-vs-back', 'kr-vs-k-one-vs-fifteen']

datasetnames_4 = ['dermatology-6', 'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8', 'wdbc',
                  'poker-8-9-vs-6', 'yeast1', 'german']

# datasetnames_4 = ['shuttle-c0-vs-c4', 'abalone-20-vs-8-9-10', 'poker-8-9-vs-5']

datasetnames_5 = ['winequality-red-8-vs-6', 'wisconsin', 'australian', 'abalone9-18', 'pima', 'vehicle2',
                  'winequality-red-8-vs-6-7', 'yeast-0-2-5-6-vs-3-7-8-9']

# datasetnames_5 = ['winequality-white-3-vs-7']

datasetnames_6 = ['page-blocks0', 'shuttle-2-vs-5']

# datasetnames = [datasetnames_1, datasetnames_2, datasetnames_3, datasetnames_4, datasetnames_5, datasetnames_6]
datasetnames = datasetnames_1 + datasetnames_2 + datasetnames_3 + datasetnames_4 + datasetnames_5 + datasetnames_6

# datasetnames_final_1 = ['appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass0', 'new-thyroid1',
#                       'shuttle-6-vs-2-3', 'heart',
#                       'spambase', 'banana', 'phoneme', 'ecoli1', 'ecoli2', 'segment0', 'led7digit-0-2-4-5-6-7-8-9-vs-1',
#                       'yeast-1-vs-7', 'yeast-2-vs-8', 'wdbc', 'yeast1', 'german', 'wisconsin', 'australian', 'pima',
#                       'vehicle2', 'winequality-red-8-vs-6-7', 'yeast-0-2-5-6-vs-3-7-8-9', "shuttle-2-vs-5"]

datasetnames_final_1 = [
    'sonar', 'banana', 'australian', 'heart', 'spambase', 'wdbc', 'wisconsin',
    'pima', 'iris0', 'glass0', 'german', 'phoneme', 'yeast1', 'vehicle2',
    'ecoli1', 'appendicitis', 'new-thyroid1', 'ecoli2', 'segment0',
    'yeast-0-2-5-6-vs-3-7-8-9', 'led7digit-0-2-4-5-6-7-8-9-vs-1',
    'cleveland-0-vs-4', 'yeast-1-vs-7', 'shuttle-6-vs-2-3', 'yeast-2-vs-8',
    'winequality-red-8-vs-6-7', 'shuttle-2-vs-5']

num_run = 3

POPSIZE = 30  # 种群大小
CXPB = 0.8  # 交叉概率
MUTPB = 0.2  # 变异概率
NGEN = 100  # 迭代次数
verbose = False  # 是否打印信息

evol_parameter = EvolutionaryParameterConfig(POPSIZE, CXPB, MUTPB, NGEN, verbose)

# 保存路径
file_path = '../../datasets/dat/'

# 表头
columns_dataset = ['F-measure', 'G-mean', 'AUC']
columns_datasets = ['数据集', '实例数量', '特征数量', 'F-measure', 'G-mean', 'AUC']

# 评价指标
scoring = {
    'f1_macro': 'f1_macro',
    'g_mean': 'g_mean',
    'roc_auc_ovr': 'roc_auc_ovr'
}
