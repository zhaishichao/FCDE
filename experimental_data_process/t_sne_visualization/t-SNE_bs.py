import os
import pandas as pd
from visualize import plot_tsne

# ======== 数据配置 ========
dataset_names = ['heart', 'wisconsin', 'pima', 'shuttle']
titles = ['D2', 'D5', 'D6', 'D25']
file_names = ['heart_1_X_train_res_bs.csv', 'wisconsin_2_X_train_res_bs.csv', 'pima_3_X_train_res_bs.csv',
              'shuttle-2-vs-5_1_X_train_res_bs.csv']
root_path = "D:\\ApplicationDoc\\Downloads\\tsne\\bs和mt\\"
save_dir = "./results/"
show_legend = False  # 是否显示图例，需要时改为 True

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for dataset_name, title, file_name in zip(dataset_names, titles, file_names):
    save_name = dataset_name + '_' + title.lower() + '_bs'
    df = pd.read_csv(os.path.join(root_path, file_name))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    plot_tsne(X, y, show_legend=show_legend, save_path=os.path.join(save_dir, save_name + ".pdf"))
