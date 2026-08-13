from common import plot_single_group

# ======== 数据配置 ========
dataset_names = ['heart', 'wisconsin', 'pima', 'shuttle']
titles = ['D2', 'D5', 'D6', 'D25']
file_names = ['heart_1_X_train_res_bs.csv', 'wisconsin_2_X_train_res_bs.csv', 'pima_3_X_train_res_bs.csv',
              'shuttle-2-vs-5_1_X_train_res_bs.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\最新结果\\tsne\\bs和mt\\"
save_dir = "./results/"

plot_single_group(
    dataset_names, titles, file_names, root_path, save_dir,
    suffix="bs",
    legend_rule=lambda t: False,
)
