from config import datasetnames_final_1, num_run
from test.experiment_runner import run_experiment

if __name__ == '__main__':
    run_experiment('knn', 'bs', datasetnames_final_1, num_run)
