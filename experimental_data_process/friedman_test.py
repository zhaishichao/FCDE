import os

import pandas as pd
import numpy as np
from scipy.stats import rankdata, friedmanchisquare

def rank_and_friedman(
    input_csv,
    output_rank_csv="rank_result.csv",
    gp_method="GP-SMOTE"
):
    """
    功能：
    1. 读取均值CSV
    2. 每个数据集内部按值排序（值越大越好）
       并列取平均名次
    3. 保存排名CSV
    4. 输出平均排名
    5. 对原始均值做 Friedman test
    """

    # ==================================
    # 读取数据
    # ==================================
    rootpath = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\mean\\lw\\knn\\"
    input_csv = os.path.join(rootpath, input_csv)
    df = pd.read_csv(input_csv)

    dataset_col = df.columns[0]          # 第一列 Dataset
    methods = list(df.columns[1:])       # 方法列

    # ==================================
    # 计算排名
    # ==================================
    rank_rows = []

    for _, row in df.iterrows():

        dataset = row[dataset_col]
        values = row[methods].values.astype(float)

        # 降序排名（值越大排名越靠前）
        ranks = rankdata(-values, method='average')

        rank_rows.append([dataset] + list(ranks))

    rank_df = pd.DataFrame(rank_rows, columns=[dataset_col] + methods)
    output_rank_csv = os.path.join(rootpath, output_rank_csv)

    # 保存排名文件
    rank_df.to_csv(output_rank_csv, index=False)

    print("排名文件已保存：", output_rank_csv)
    print(rank_df)

    # ==================================
    # 平均排名
    # ==================================
    avg_rank = rank_df[methods].mean(axis=0)

    avg_rank_df = pd.DataFrame({
        "Method": methods,
        "Average Rank": avg_rank.values
    }).sort_values(by="Average Rank")

    print("\n平均排名（越小越好）:")
    print(avg_rank_df)

    # ==================================
    # Friedman test
    # ==================================
    data = [df[m].values for m in methods]

    stat, p = friedmanchisquare(*data)

    print("\n========== Friedman Test ==========")
    print("Statistic =", round(stat, 6))
    print("p-value   =", round(p, 6))

    if p < 0.05:
        print("存在显著性差异（p < 0.05）")
    else:
        print("不存在显著性差异（p >= 0.05）")

    # ==================================
    # GP-SMOTE 与其他算法平均排名对比
    # ==================================
    print("\n========== GP-SMOTE 平均排名对比 ==========")

    gp_rank = avg_rank[gp_method]

    for m in methods:
        if m == gp_method:
            continue

        diff = avg_rank[m] - gp_rank

        if diff > 0:
            result = "GP-SMOTE更优"
        elif diff < 0:
            result = "GP-SMOTE更差"
        else:
            result = "相同"

        print(f"{gp_method} vs {m}: {result}")

    return rank_df, avg_rank_df


# ==========================================
# 示例运行
# ==========================================
# rank_and_friedman(
#     input_csv="f1_1.csv",
#     output_rank_csv="f1_1_rank.csv",
#     gp_method="GP-SMOTE"
# )
# rank_and_friedman(
#     input_csv="auc_1.csv",
#     output_rank_csv="auc_1_rank.csv",
#     gp_method="GP-SMOTE"
# )
# rank_and_friedman(
#     input_csv="f1_2.csv",
#     output_rank_csv="f1_2_rank.csv",
#     gp_method="GP-SMOTE"
# )
# rank_and_friedman(
#     input_csv="auc_2.csv",
#     output_rank_csv="auc_2_rank.csv",
#     gp_method="GP-SMOTE"
# )
rank_and_friedman(
    input_csv="runtime.csv",
    output_rank_csv="runtime_rank.csv",
    gp_method="GP-SMOTE"
)