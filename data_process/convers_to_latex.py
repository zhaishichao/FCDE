import pandas as pd

# 需要筛选的数据集
datasetnames = [
    'appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass0', 
    'new-thyroid1', 'shuttle-6-vs-2-3', 'heart', 'spambase', 'banana', 
    'phoneme', 'ecoli1', 'ecoli2', 'segment0', 
    'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8', 
    'wdbc', 'yeast1', 'german', 'wisconsin', 'australian', 'pima', 
    'vehicle2', 'winequality-red-8-vs-6-7', 
    'yeast-0-2-5-6-vs-3-7-8-9', "shuttle-2-vs-5"
]

# 读取统计文件
df = pd.read_csv("../datasets/dataset_statistics.csv")

# 筛选数据集
df_filtered = df[df["dataset"].isin(datasetnames)].copy()

# 按 imbalance_ratio 升序排序（关键步骤）
df_filtered = df_filtered.sort_values(by="imbalance_ratio", ascending=True)

# 添加 Index 列（D1, D2, ...）
df_filtered["Index"] = [f"D{i+1}" for i in range(len(df_filtered))]

# 调整列顺序
df_filtered = df_filtered[["dataset", "Index", "n_features", "n_samples", "imbalance_ratio"]]

# 修改列名
df_filtered.columns = ["Dataset", "Index", "#F.", "#Inst.", "IR"]

# 保存 CSV
df_filtered.to_csv("../datasets/sorted_dataset_statistics.csv", index=False)

print("Saved CSV: sorted_dataset_statistics.csv")

latex_table = df_filtered.to_latex(
    index=False,
    column_format="ccccc",
    caption="Statistics of selected datasets sorted by imbalance ratio.",
    label="tab:dataset_statistics",
    float_format="%.3f"
)

print(latex_table)