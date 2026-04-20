import os.path
import re
import pandas as pd

def latex_to_csv(raw_text, columns, save_path="output.csv"):
    """
    功能：
    1. 读取多行 LaTeX 表格代码
    2. 每行提取：数据集名称 + 各列均值（只取括号前数字）
    3. 保存为 CSV

    参数：
    raw_text   : 多行字符串（latex代码）
    columns    : DataFrame表头（你自定义）
    save_path  : 保存路径
    """

    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    rows = []

    for line in lines:
        # 去掉行尾 \\
        line = line.replace(r"\\", "").strip()

        # 按 & 分列
        parts = [x.strip() for x in line.split("&")]

        # 第一列：数据集名称
        dataset = parts[0]

        values = []

        for item in parts[1:]:
            # 去掉 \hl{}
            item = item.replace(r"\hl{", "").replace("}", "")

            # 提取第一个数字（均值）
            m = re.search(r'(\d+\.\d+)', item)
            if m:
                values.append(float(m.group(1)))

        rows.append([dataset] + values)

    # 构造 DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # 保存 CSV
    rootpath = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\mean\\lw\\knn\\"
    save_path = os.path.join(rootpath, save_path)

    df.to_csv(save_path, index=False)

    print("保存成功：", save_path)
    print(df)

    return df


# ==========================================
# 示例使用
# ==========================================
raw_text = r"""
D1 & 56.14 & 71.23 & 67.25 & \textbf{9.96} \\
D2 & 17.20 & 25.68 & 25.93 & \textbf{2.98} \\
D3 & 2504.05 & 999.99 & 845.94 & \textbf{791.93} \\
D4 & 76.62 & 128.43 & 50.35 & \textbf{15.82} \\
D5 & 158.20 & 181.12 & 58.18 & \textbf{29.80} \\
D6 & 161.66 & 207.05 & 65.82 & \textbf{34.54} \\
D7 & 23.16 & 42.82 & 17.94 & \textbf{3.36} \\
D8 & 28.91 & 60.11 & 21.51 & \textbf{4.40} \\
D9 & 271.40 & 265.00 & 64.43 & \textbf{72.21} \\
D10 & 3960.55 & 827.39 & \textbf{601.07} & 1179.85 \\
D11 & 349.86 & 213.75 & 141.68 & \textbf{98.55} \\
D12 & 148.08 & 107.58 & 58.33 & \textbf{39.05} \\
D13 & 50.23 & 73.57 & 30.31 & \textbf{8.90} \\
D14 & 18.32 & 18.79 & 14.30 & \textbf{1.83} \\
D15 & 44.90 & 54.39 & 20.68 & \textbf{4.30} \\
D16 & 58.37 & 90.85 & 29.10 & \textbf{10.54} \\
D17 & 744.76 & 779.34 & 426.86 & \textbf{352.99} \\
D18 & 270.64 & 378.01 & 86.04 & \textbf{82.34} \\
D19 & 134.11 & 169.08 & 35.82 & \textbf{19.59} \\
D20 & 45.65 & 69.44 & 18.19 & \textbf{3.99} \\
D21 & 113.53 & 182.61 & 36.18 & \textbf{20.54} \\
D22 & 58.83 & 101.98 & 21.73 & \textbf{6.54} \\
D23 & 133.24 & 208.78 & 38.05 & \textbf{26.88} \\
D24 & 214.11 & 381.84 & 69.80 & \textbf{67.48} \\
D25 & 799.78 & 1593.31 & \textbf{368.73} & 435.10 \\
"""

# 你自己定义表头
# columns = [
#     "Dataset",
#     "RAW",
#     "ROS",
#     "SMOTE",
#     "KMeans-SMOTE",
#     "Borderline_1",
#     "Borderline_2",
#     "GP-SMOTE"
# ]
columns = [
    "Dataset",
    "DG-SMOTE",
    "MTGP-SMOTE",
    "Blind-SMOTE",
    "GP-SMOTE"
]
latex_to_csv(raw_text, columns, save_path="runtime.csv")