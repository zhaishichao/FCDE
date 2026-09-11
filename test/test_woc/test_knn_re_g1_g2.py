from config import datasetnames
from test.experiment_runner import run_experiment

if __name__ == '__main__':
    run_experiment('knn', 'gp', datasetnames, n_runs=2,
                   sampler_params=dict(pop_size=30, cx_prob=0.8,
                                       mut_prob=0.2, n_gen=100, verbose=False,
                                       remove_constraints=(1, 2)))
