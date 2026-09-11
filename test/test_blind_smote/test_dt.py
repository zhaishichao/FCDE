from config import datasetnames
from test.experiment_runner import run_experiment

if __name__ == '__main__':
    run_experiment('dt', 'bs', datasetnames, n_runs=15,
                   sampler_params=dict(pop_size=30, n_gen=30))
