from config import datasetnames_final_1
from test.experiment_runner import run_experiment

if __name__ == '__main__':
    run_experiment('knn', 'dg', datasetnames_final_1, n_runs=2,
                   sampler_params=dict(pop_size=30, cx_prob=0.8,
                                       mut_prob=0.2, n_gen=100, verbose=False))
