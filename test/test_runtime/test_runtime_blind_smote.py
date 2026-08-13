from smote_variants.blind_smote.blind_smote import BlindSMOTE
from test.test_runtime.runtime_benchmark import benchmark

num_run = 3
POP_SIZE = 100
N_GEN = 500


def run_once(X_train, y_train, seed):
    blind = BlindSMOTE(
        k=5,
        N_min=1,
        N_max=10,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        cx_prob=0.8,
        mut_prob=0.05,
        mut_bit_rate=0.01,
        elitism_ratio=0.1,
        stagnation_gens=100,
        random_state=seed)
    blind.fit_resample(X_train, y_train)


if __name__ == '__main__':
    benchmark("BlindSMOTE", run_once, save_dir="blind_smote", num_run=num_run)
