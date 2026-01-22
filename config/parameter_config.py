class EvolutionaryParameterConfig:
    def __init__(self, POPSIZE=30, CXPB=0.9, MUTPB=0.1, NGEN=50, verbose=True):
        self.POPSIZE = POPSIZE  # 种群大小
        self.CXPB = CXPB  # 交叉概率
        self.MUTPB = MUTPB  # 变异概率
        self.NGEN = NGEN  # 迭代次数
        self.verbose = verbose  # 是否打印信息


datasetnames = ['twonorm', 'ring', 'chess', 'sonar', 'banana', 'australian', 'heart', 'spambase', 'wdbc', 'ionosphere',
                'glass1', 'magic', 'wisconsin', 'pima', 'iris0', 'glass0', 'titanic', 'german', 'phoneme', 'yeast1',
                'haberman', 'vehicle2', 'adult', 'ecoli1', 'appendicitis', 'new-thyroid1', 'ecoli2', 'segment0',
                'ecoli3', 'page-blocks0', 'yeast-0-2-5-6-vs-3-7-8-9', 'vowel0', 'led7digit-0-2-4-5-6-7-8-9-vs-1',
                'cleveland-0-vs-4', 'ecoli-0-1-4-6-vs-5', 'shuttle-c0-vs-c4', 'yeast-1-vs-7', 'coil2000', 'abalone9-18',
                'dermatology-6', 'shuttle-6-vs-2-3', 'yeast-2-vs-8', 'flare-F', 'car-good', 'kr-vs-k-one-vs-fifteen',
                'yeast5', 'winequality-red-8-vs-6', 'winequality-white-3-vs-7', 'winequality-red-8-vs-6-7',
                'poker-8-9-vs-6', 'shuttle-2-vs-5', 'abalone-20-vs-8-9-10', 'poker-8-9-vs-5',
                'kddcup-rootkit-imap-vs-back']

datasetnames_instances = ['appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass1', 'glass0', 'new-thyroid1',
                'shuttle-6-vs-2-3', 'heart', 'ecoli-0-1-4-6-vs-5', 'haberman', 'ecoli1', 'ecoli2', 'ecoli3',
                'ionosphere', 'dermatology-6', 'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8', 'wdbc',
                'winequality-red-8-vs-6', 'wisconsin', 'australian', 'abalone9-18', 'pima', 'vehicle2',
                'winequality-red-8-vs-6-7', 'winequality-white-3-vs-7', 'vowel0', 'german', 'yeast-0-2-5-6-vs-3-7-8-9',
                'flare-F', 'yeast5', 'yeast1', 'poker-8-9-vs-6', 'car-good', 'shuttle-c0-vs-c4', 'abalone-20-vs-8-9-10',
                'poker-8-9-vs-5', 'titanic', 'kddcup-rootkit-imap-vs-back', 'kr-vs-k-one-vs-fifteen', 'segment0',
                'chess', 'shuttle-2-vs-5', 'spambase', 'banana', 'phoneme', 'page-blocks0', 'twonorm', 'ring',
                'coil2000', 'magic', 'adult']

datasetnames_1 = ['appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass1', 'twonorm', 'ring', 'coil2000',
                  'magic', 'adult']

datasetnames_2 = ['glass0', 'new-thyroid1', 'shuttle-6-vs-2-3', 'heart', 'ecoli-0-1-4-6-vs-5', 'shuttle-2-vs-5',
                  'spambase', 'banana', 'phoneme', 'page-blocks0']

datasetnames_3 = ['haberman', 'ecoli1', 'ecoli2', 'ecoli3', 'ionosphere', 'titanic', 'kddcup-rootkit-imap-vs-back',
                  'kr-vs-k-one-vs-fifteen', 'segment0', 'chess']

datasetnames_4 = ['dermatology-6', 'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8', 'wdbc',
                  'poker-8-9-vs-6', 'car-good', 'shuttle-c0-vs-c4', 'abalone-20-vs-8-9-10', 'poker-8-9-vs-5']

datasetnames_5 = ['winequality-red-8-vs-6', 'wisconsin', 'australian', 'abalone9-18', 'pima', 'vehicle2',
                'winequality-red-8-vs-6-7', 'winequality-white-3-vs-7', 'vowel0', 'german', 'yeast-0-2-5-6-vs-3-7-8-9',
                'flare-F', 'yeast5', 'yeast1']
