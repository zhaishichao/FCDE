import os
import pandas as pd

# 五个文件夹的路径列表
folders = [
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\汇总结果\\0",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\汇总结果\\1",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\汇总结果\\2",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\汇总结果\\3-SVM缺一个",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\汇总结果\\4",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\汇总结果\\5-SVM缺少一个"
]

# 三个分类器
classifiers = ["dt", "knn", "svm"]

# 算法对应的文件名
algorithms = {
    "DG-SMOTE": "mean_dg.csv",
    "GP-SMOTE": "mean_ds.csv"
}

# 输出文件夹
output_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\最终对比\\"
os.makedirs(output_root, exist_ok=True)

# 遍历每个分类器
for clf in classifiers:
    clf_output_dir = os.path.join(output_root, clf)
    os.makedirs(clf_output_dir, exist_ok=True)

    for algo_name, file_name in algorithms.items():
        combined_df = pd.DataFrame()

        # 遍历五个文件夹，读取对应分类器下的算法结果
        for folder in folders:
            file_path = os.path.join(folder, clf, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"Warning: {file_path} does not exist!")

        # 保存合并后的文件
        output_file = os.path.join(clf_output_dir, f"{algo_name}.csv")
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Saved merged file: {output_file}")
