import os
import pandas as pd

# 分类器
classifiers = ["dt", "knn", "svm"]

# 对比方法（7种）
methods = [
    "RAW",
    "ROS",
    "SMOTE",
    "SMOTEN",
    "Borderline_1",
    "Borderline_2",
    "DG-SMOTE"
]

metrics = ["F-measure", "G-mean", "AUC"]

root_dir = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\最终对比\\"  # svm/knn/dt 所在目录
output_dir = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\最终对比\\"
os.makedirs(output_dir, exist_ok=True)

for clf in classifiers:
    clf_path = os.path.join(root_dir, clf)

    # 读取 GP-SMOTE
    gp_file = os.path.join(clf_path, "GP-SMOTE.csv")
    df_gp = pd.read_csv(gp_file)

    dataset_col = df_gp.columns[0]

    result_rows = []

    # 遍历每个数据集（按行）
    for i in range(len(df_gp)):
        row_result = {}
        row_result["数据集"] = df_gp.iloc[i][dataset_col]

        count_list = []  # 存每个metric的胜出次数

        # 对每个metric处理
        for metric in metrics:
            compare_array = []

            for method in methods:
                method_file = os.path.join(clf_path, f"{method}.csv")

                if not os.path.exists(method_file):
                    print(f"Missing: {method_file}")
                    compare_array.append(0)
                    continue

                df_method = pd.read_csv(method_file)

                gp_val = df_gp.iloc[i][metric]
                method_val = df_method.iloc[i][metric]

                compare_array.append(1 if gp_val > method_val else 0)

            row_result[metric] = compare_array
            count_list.append(sum(compare_array))

        row_result["个数"] = count_list
        result_rows.append(row_result)

    # 转为 DataFrame
    result_df = pd.DataFrame(result_rows)

    # 保存
    output_file = os.path.join(output_dir, f"{clf}_gp_compare.csv")
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Saved: {output_file}")
