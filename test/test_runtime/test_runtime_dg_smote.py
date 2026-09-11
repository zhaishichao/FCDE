from smote_variants.dg_smote import DGSMOTE
from config import EvolutionaryParameterConfig
from test.test_runtime.runtime_benchmark import benchmark

num_run = 10
POPSIZE = 30
CXPB = 0.8
MUTPB = 0.2
NGEN = 100

evol_parameter = EvolutionaryParameterConfig(POPSIZE, CXPB, MUTPB, NGEN, verbose=False)


def run_once(X_train, y_train, seed):
    dg = DGSMOTE(pop_size=evol_parameter.POPSIZE, cx_prob=evol_parameter.CXPB, mut_prob=evol_parameter.MUTPB, n_gen=evol_parameter.NGEN, verbose=evol_parameter.verbose)
    dg.fit_resample(X_train, y_train)


if __name__ == '__main__':
    benchmark("DGSMOTE", run_once, save_dir="dg_smote", num_run=num_run)
