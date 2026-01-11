# 实验参数设置
from config import EvolutionaryParameterConfig

datasetnames = ['iris0', 'ecoli1', 'glass0', 'glass1', 'haberman', 'pima', 'segment0', 'vowel0', 'wisconsin',
                'yeast1', 'shuttle-c0-vs-c4', 'australian']
num_run = 1

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
