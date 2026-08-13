from smote_variants.gp_smote_population_div_random import DSSMOTE
from config import EvolutionaryParameterConfig
from test.test_runtime.runtime_benchmark import benchmark

num_run = 3
POPSIZE = 30
CXPB = 0.8
MUTPB = 0.2
NGEN = 100

evol_parameter = EvolutionaryParameterConfig(POPSIZE, CXPB, MUTPB, NGEN, verbose=False)


def run_once(X_train, y_train, seed):
    gp = DSSMOTE(X=X_train, y=y_train, evol_parameter=evol_parameter)
    gp.fit_resample()


if __name__ == '__main__':
    benchmark("GPSMOTE", run_once, save_dir="gp_smote", num_run=num_run)
