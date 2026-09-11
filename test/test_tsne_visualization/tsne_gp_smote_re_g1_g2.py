from test.test_tsne_visualization.common import run_tsne
from smote_variants.gp_smote import GPSMOTE


def make_sampler(seed):
    return GPSMOTE(pop_size=30, cx_prob=0.8, mut_prob=0.2, n_gen=100,
                   verbose=False, remove_constraints=(1, 2),
                   res_only=True, random_state=seed)


def extract_synthetic(sampler, X, y):
    return sampler.fit_resample(X, y)[0]  # res_only=True 返回 (X_syn, y_syn)


if __name__ == '__main__':
    run_tsne('gp_smote_re_g1_g2', make_sampler, extract_synthetic)
