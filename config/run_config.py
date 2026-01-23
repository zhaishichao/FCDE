# 实验参数设置
from config import EvolutionaryParameterConfig

datasetnames = ['appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass1', 'glass0', 'new-thyroid1',
                'shuttle-6-vs-2-3', 'heart', 'ecoli-0-1-4-6-vs-5', 'haberman', 'ecoli1', 'ecoli2', 'ecoli3',
                'ionosphere', 'dermatology-6', 'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8', 'wdbc',
                'winequality-red-8-vs-6', 'wisconsin', 'australian', 'abalone9-18', 'pima', 'vehicle2',
                'winequality-red-8-vs-6-7', 'winequality-white-3-vs-7', 'vowel0', 'german', 'yeast-0-2-5-6-vs-3-7-8-9',
                'flare-F', 'yeast5', 'yeast1', 'poker-8-9-vs-6', 'car-good', 'shuttle-c0-vs-c4', 'abalone-20-vs-8-9-10',
                'poker-8-9-vs-5', 'titanic', 'kddcup-rootkit-imap-vs-back', 'kr-vs-k-one-vs-fifteen', 'segment0',
                'chess', 'shuttle-2-vs-5', 'spambase', 'banana', 'phoneme', 'page-blocks0', 'twonorm', 'ring',
                'coil2000', 'magic', 'adult']

datasetnames_1 = ['appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass1', 'ring', 'magic']

datasetnames_2 = ['glass0', 'new-thyroid1', 'shuttle-6-vs-2-3', 'heart', 'ecoli-0-1-4-6-vs-5', 'spambase', 'banana',
                  'phoneme']

datasetnames_3 = ['haberman', 'ecoli1', 'ecoli2', 'ecoli3', 'ionosphere', 'titanic', 'kddcup-rootkit-imap-vs-back',
                  'kr-vs-k-one-vs-fifteen', 'segment0', 'chess']

datasetnames_4 = ['dermatology-6', 'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8', 'wdbc',
                  'poker-8-9-vs-6', 'car-good', 'shuttle-c0-vs-c4', 'abalone-20-vs-8-9-10', 'poker-8-9-vs-5']

datasetnames_5 = ['winequality-red-8-vs-6', 'wisconsin', 'australian', 'abalone9-18', 'pima', 'vehicle2',
                  'winequality-red-8-vs-6-7', 'winequality-white-3-vs-7', 'vowel0', 'german',
                  'yeast-0-2-5-6-vs-3-7-8-9', 'flare-F', 'yeast5', 'yeast1']

datasetnames_6 = ['page-blocks0', 'shuttle-2-vs-5', 'coil2000']

num_run = 30

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
