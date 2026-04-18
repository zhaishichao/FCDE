import os
import pandas as pd

# 分类器
classifiers = ["dt", "knn", "svm"]

# 方法（注意文件名要和你的实际一致）
methods = [
    "Borderline_1",
    "Borderline_2",
    "ROS",
    "Raw",
    "SMOTE",
    "SMOTEN",
    "DG-SMOTE",
    "Blind-SMOTE",
    "MTGP-SMOTE",
    "GP-SMOTE"
]


# 指定要提取的数据集
datasetnames = [
    'appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass0',
    'new-thyroid1', 'shuttle-6-vs-2-3', 'heart', 'spambase', 'banana',
    'phoneme', 'ecoli1', 'ecoli2', 'segment0',
    'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8',
    'wdbc', 'yeast1', 'german', 'wisconsin', 'australian', 'pima',
    'vehicle2', 'winequality-red-8-vs-6-7',
    'yeast-0-2-5-6-vs-3-7-8-9', "shuttle-2-vs-5"
]

# 只提取的指标
metrics = ["F-measure", "AUC"]

root_dir = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\GP_vs_All\\"  # dt/knn/svm 所在目录
output_dir = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\F1和AUC\\"
os.makedirs(output_dir, exist_ok=True)

for clf in classifiers:
    clf_path = os.path.join(root_dir, clf)
    clf_output = os.path.join(output_dir, clf)
    os.makedirs(clf_output, exist_ok=True)

    for method in methods:
        file_path = os.path.join(clf_path, f"{method}.csv")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # 默认第一列是数据集名称（如果不是请修改）
        dataset_col = df.columns[0]

        # 筛选指定数据集
        df_filtered = df[df[dataset_col].isin(datasetnames)]

        # 只保留 数据集 + 指标列
        df_filtered = df_filtered[[dataset_col] + metrics]

        # 保存
        output_file = os.path.join(clf_output, f"{method}.csv")
        df_filtered.to_csv(output_file, index=False, float_format="%.4f", encoding="utf-8-sig")

        print(f"Saved: {output_file}")
