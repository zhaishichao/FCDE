import os
import pandas as pd
from visualize.tsne_plot import plot_tsne

# ======== 数据配置 ========
dataset_names = ['heart', 'wisconsin', 'pima', 'shuttle']
titles = ['D2', 'D5', 'D6', 'D25']
file_names = ['heart_4_X_train_res_dg.csv', 'wisconsin_1_X_train_res_dg.csv', 'pima_4_X_train_res_dg.csv',
              'shuttle-2-vs-5_1_X_train_res_ds.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\ds和dg\\"
save_dir = "./results/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for dataset_name, title, file_name in zip(dataset_names, titles, file_names):
    save_name = dataset_name + '_' + title.lower() + '_gp'
    show_legend = (title == 'D4')
    df = pd.read_csv(os.path.join(root_path, file_name))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    plot_tsne(X, y, show_legend=show_legend, save_path=os.path.join(save_dir, save_name + ".pdf"))
