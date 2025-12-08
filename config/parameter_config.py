class EvolutionaryParameterConfig:
    def __init__(self, POPSIZE=30, CXPB=0.9, MUTPB=0.1, NGEN=50, verbose=True):
        self.POPSIZE = POPSIZE  # 种群大小
        self.CXPB = CXPB  # 交叉概率
        self.MUTPB = MUTPB  # 变异概率
        self.NGEN = NGEN  # 迭代次数
        self.verbose = verbose  # 是否打印信息


datasetnames = ['iris0', 'ecoli1', 'glass0', 'glass1', 'haberman', 'pima', 'segment0', 'vowel0', 'wisconsin', 'yeast1',
                'shuttle-c0-vs-c4']
datasetnames = ['australian', 'heart', 'phoneme', 'ring', 'spambase', 'wdbc', 'ionosphere']
