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
    dg = DGSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
    dg.fit_resample()


if __name__ == '__main__':
    benchmark("DGSMOTE", run_once, save_dir="dg_smote", num_run=num_run)
