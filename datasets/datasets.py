import os
import numpy as np
import pandas as pd

from data_preprocess import data_loader
def analyze_binary_datasets(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.dat'):
            continue

        file_path = os.path.join(folder_path, file_name)
        dataset_name = os.path.splitext(file_name)[0]

        # data = load_dat_file(file_path)

        X,y = data_loader(file_path)
        # 实例数 & 特征数
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # 统计类别分布
        unique_labels, counts = np.unique(y, return_counts=True)

        if len(unique_labels) != 2:
            raise ValueError(f"{dataset_name} 不是二分类数据集")

        majority = counts.max()
        minority = counts.min()
        imbalance_ratio = majority / minority

        results.append({
            "dataset": dataset_name,
            "n_samples": n_samples,
            "n_features": n_features,
            "imbalance_ratio": imbalance_ratio
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    folder_path = "dat/"
    df = analyze_binary_datasets(folder_path)
    df = df.sort_values(by="n_samples", ascending=True)
    df.to_csv("dataset_statistics.csv", index=False)
    dataset_names = df["dataset"].tolist()
    print(df)
    print(dataset_names)


