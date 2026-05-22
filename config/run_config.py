# 实验参数设置
from config import EvolutionaryParameterConfig

datasetnames_final_1 = ['australian', 'heart', 'spambase', 'wdbc', 'wisconsin',
                        'pima', 'iris0', 'glass0', 'german', 'phoneme', 'yeast1', 'vehicle2',
                        'ecoli1', 'appendicitis', 'new-thyroid1', 'ecoli2', 'segment0',
                        'yeast-0-2-5-6-vs-3-7-8-9', 'led7digit-0-2-4-5-6-7-8-9-vs-1',
                        'cleveland-0-vs-4', 'yeast-1-vs-7', 'shuttle-6-vs-2-3', 'yeast-2-vs-8',
                        'winequality-red-8-vs-6-7', 'shuttle-2-vs-5']

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
