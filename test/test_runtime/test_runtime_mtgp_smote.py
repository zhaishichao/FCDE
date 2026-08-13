from smote_variants.mtgp_smote.mtgp_smote import MTGPSMOTESampler
from test.test_runtime.runtime_benchmark import benchmark

num_run = 10
POP_SIZE = 30
N_GENERATIONS = 100


def run_once(X_train, y_train, seed):
    mtgp = MTGPSMOTESampler(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        cx_rate=0.7,
        mut_rate=0.3,
        tournament_k=3,
        max_depth=6,
        random_state=seed)
    mtgp.fit_resample(X_train, y_train)


if __name__ == '__main__':
    benchmark("MTGPSMOTE", run_once, save_dir="mtgp_smote", num_run=num_run)
