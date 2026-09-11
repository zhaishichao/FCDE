from test.test_tsne_visualization.common import run_tsne
from smote_variants.blind_smote import BlindSMOTE


def make_sampler(seed):
    return BlindSMOTE(pop_size=30, n_gen=30, res_only=True, random_state=seed)


def extract_synthetic(sampler, X, y):
    return sampler.fit_resample(X, y)[2]  # res_only=True 返回 (X_res, y_res, synth_rows)


if __name__ == '__main__':
    run_tsne('blind_smote', make_sampler, extract_synthetic)
