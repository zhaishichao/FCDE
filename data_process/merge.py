import os
import shutil

# =============================
# 1. 五个主目录（按你的实际路径修改）
# =============================
source_roots = [
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\\gp、dg\\汇总结果\\0",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\gp、dg\\汇总结果\\1",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\\gp、dg\\汇总结果\\2",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\\gp、dg\\汇总结果\\3-SVM缺一个",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\\gp、dg\\汇总结果\\4",
    "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\\gp、dg\\汇总结果\\5-SVM缺少一个"
]

# =============================
# 2. 新目录（输出目录）
# =============================
target_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\dggp\\"

# =============================
# 3. 需要提取的27个数据集（示例）
# =============================
dataset_list = [
    'appendicitis', 'iris0', 'cleveland-0-vs-4', 'sonar', 'glass0',
    'new-thyroid1', 'shuttle-6-vs-2-3', 'heart', 'spambase', 'banana',
    'phoneme', 'ecoli1', 'ecoli2', 'segment0',
    'led7digit-0-2-4-5-6-7-8-9-vs-1', 'yeast-1-vs-7', 'yeast-2-vs-8',
    'wdbc', 'yeast1', 'german', 'wisconsin', 'australian', 'pima',
    'vehicle2', 'winequality-red-8-vs-6-7',
    'yeast-0-2-5-6-vs-3-7-8-9', "shuttle-2-vs-5"
]

# =============================
# 4. 分类器与方法映射
# =============================
classifiers = ["dt", "knn", "svm"]

# 原目录名 -> 新目录名
methods_map = {
    "dg": "DG-SMOTE",
    "ds": "GP-SMOTE"
}

# =============================
# 5. 创建目标目录结构
# =============================
for clf in classifiers:
    for new_method in methods_map.values():
        save_dir = os.path.join(target_root, clf, new_method)
        os.makedirs(save_dir, exist_ok=True)

# =============================
# 6. 开始遍历复制
# =============================
for clf in classifiers:
    for old_method, new_method in methods_map.items():
        target_dir = os.path.join(target_root, clf, new_method)

        for root in source_roots:
            source_dir = os.path.join(root, clf, old_method)

            if not os.path.exists(source_dir):
                continue

            for dataset in dataset_list:
                file_name = dataset + ".csv"
                source_file = os.path.join(source_dir, file_name)

                if os.path.exists(source_file):
                    target_file = os.path.join(target_dir, file_name)

                    # 若已存在则跳过（避免重复覆盖）
                    if not os.path.exists(target_file):
                        shutil.copy2(source_file, target_file)
                        print(f"已复制: {source_file} -> {target_file}")

print("全部完成！")