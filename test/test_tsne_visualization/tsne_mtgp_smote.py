from test.test_tsne_visualization.common import run_tsne
from smote_variants.mtgp_smote.mtgp_smote import MTGPSMOTESampler


def make_sampler(seed):
    return MTGPSMOTESampler(pop_size=30, n_generations=100, cx_rate=0.7, mut_rate=0.3,
                            tournament_k=3, max_depth=4, res_only=True, random_state=seed)


def extract_synthetic(sampler, X, y):
    return sampler.fit_resample(X, y)  # res_only=True 返回 X_syn


if __name__ == '__main__':
    run_tsne('mtgp_smote', make_sampler, extract_synthetic)
