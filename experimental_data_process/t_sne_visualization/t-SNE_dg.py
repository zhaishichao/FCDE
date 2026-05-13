import os
import pandas as pd
from visualize.tsne_plot import plot_tsne

# ======== 数据配置 ========
dataset_names = ['heart', 'wisconsin', 'pima', 'shuttle']
titles = ['D2', 'D5', 'D6', 'D25']
file_names = ['heart_3_X_train_res_ds.csv', 'wisconsin_5_X_train_res_ds.csv', 'pima_4_X_train_res_ds.csv',
              'shuttle-2-vs-5_3_X_train_res_dg.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\"
save_dir = "./results/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for dataset_name, title, file_name in zip(dataset_names, titles, file_names):
    save_name = dataset_name + '_' + title.lower() + '_dg'
    show_legend = title in ('D5', 'D6')
    df = pd.read_csv(os.path.join(root_path, file_name))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    plot_tsne(X, y, show_legend=show_legend, save_path=os.path.join(save_dir, save_name + ".pdf"))
