class EvolutionaryParameterConfig:
    def __init__(self, POPSIZE=30, CXPB=0.9, MUTPB=0.1, NGEN=50, verbose=True):
        self.POPSIZE = POPSIZE  # 种群大小
        self.CXPB = CXPB  # 交叉概率
        self.MUTPB = MUTPB  # 变异概率
        self.NGEN = NGEN  # 迭代次数
        self.verbose = verbose  # 是否打印信息


datasetnames = ['iris0', 'ecoli1', 'glass0', 'glass1', 'haberman', 'pima', 'segment0', 'vowel0', 'wisconsin', 'yeast1',
                'shuttle-c0-vs-c4', 'australian', 'heart', 'phoneme', 'ring', 'spambase', 'wdbc', 'vehicle2',
                'abalone9-18', 'cleveland-0_vs_4', 'led7digit-0-2-4-5-6-7-8-9_vs_1', 'new-thyroid1', 'page-blocks0']
