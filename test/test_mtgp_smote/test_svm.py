from config import datasetnames
from test.experiment_runner import run_experiment

if __name__ == '__main__':
    run_experiment('svm', 'mtgp', datasetnames, n_runs=15,
                   sampler_params=dict(pop_size=30, n_generations=100,
                                       cx_rate=0.7, mut_rate=0.3,
                                       tournament_k=3, max_depth=4))
