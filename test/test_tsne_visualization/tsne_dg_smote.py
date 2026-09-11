from test.test_tsne_visualization.common import run_tsne
from smote_variants.dg_smote import DGSMOTE


def make_sampler(seed):
    return DGSMOTE(pop_size=20, cx_prob=0.8, mut_prob=0.2, n_gen=10,
                   verbose=False, res_only=True, random_state=seed)


def extract_synthetic(sampler, X, y):
    return sampler.fit_resample(X, y)[0]  # res_only=True 返回 (X_syn, y_syn)


if __name__ == '__main__':
    run_tsne('dg_smote', make_sampler, extract_synthetic)
