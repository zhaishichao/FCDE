import os
import pandas as pd
import numpy as np

# ======================
# 配置
# ======================
data_root = r"F:\gp_smote_population_div_random\knn\gp"

datasetnames = [
    "australian", "heart", "spambase", "wdbc", "wisconsin", "pima",
    "iris0", "glass0", "german", "phoneme", "yeast1", "vehicle2", "ecoli1", "appendicitis",
    "new-thyroid1", "ecoli2", "segment0", "yeast-0-2-5-6-vs-3-7-8-9",
    "led7digit-0-2-4-5-6-7-8-9-vs-1", "cleveland-0-vs-4", "yeast-1-vs-7",
    "shuttle-6-vs-2-3", "yeast-2-vs-8", "winequality-red-8-vs-6-7", "shuttle-2-vs-5"
]

metrics = ["F-measure", "AUC"]


def format_std(std):
    if std == 0:
        return "0.00"
    exp = int(np.floor(np.log10(std)))
    base = std / (10 ** exp)
    return f"{base:.2f}e{exp}"


def format_mean(val):
    return f"{100 * val:.2f}"


def load_data(dataset):
    filepath = os.path.join(data_root, f"{dataset}.csv")
    df = pd.read_csv(filepath)
    return df


def compute_stats(dataset):
    df = load_data(dataset)
    results = {}
    for metric in metrics:
        vals = df[metric].values
        results[metric] = {
            "mean": np.mean(vals),
            "std": np.std(vals, ddof=1)
        }
    return results


def main():
    rows = []
    for ds in datasetnames:
        stats = compute_stats(ds)
        row = {"Dataset": ds}
        for metric in metrics:
            row[f"{metric}_mean"] = format_mean(stats[metric]["mean"])
            row[f"{metric}_std"] = format_std(stats[metric]["std"])
        rows.append(row)

    result_df = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(__file__), "gp_smote_knn_summary.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存至: {output_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
