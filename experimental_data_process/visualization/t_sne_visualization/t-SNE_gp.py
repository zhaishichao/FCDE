from common import plot_single_group

# ======== 数据配置 ========
dataset_names = ['heart', 'wisconsin', 'pima', 'shuttle']
titles = ['D2', 'D5', 'D6', 'D25']
file_names = ['heart_4_X_train_res_dg.csv', 'wisconsin_1_X_train_res_dg.csv', 'pima_4_X_train_res_dg.csv',
              'shuttle-2-vs-5_1_X_train_res_ds.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\ds和dg\\"
save_dir = "./results/"

plot_single_group(
    dataset_names, titles, file_names, root_path, save_dir,
    suffix="gp",
    legend_rule=lambda t: t == 'D4',
)
