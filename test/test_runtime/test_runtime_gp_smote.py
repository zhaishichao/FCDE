from smote_variants.gp_smote import GPSMOTE
from test.test_runtime.runtime_benchmark import benchmark

num_run = 3
POPSIZE = 30
CXPB = 0.8
MUTPB = 0.2
NGEN = 100


def run_once(X_train, y_train, seed):
    gp = GPSMOTE(pop_size=POPSIZE, cx_prob=CXPB, mut_prob=MUTPB,
                 n_gen=NGEN, verbose=False)
    gp.fit_resample(X_train, y_train)


if __name__ == '__main__':
    benchmark("GPSMOTE", run_once, save_dir="gp_smote", num_run=num_run)
