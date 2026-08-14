from common import plot_single_group

# ======== 数据配置 ========
dataset_names = ['heart', 'wisconsin', 'pima', 'shuttle']
titles = ['D2', 'D5', 'D6', 'D25']
file_names = ['heart_4_X_train_res_ds.csv', 'wisconsin_2_X_train_res_ds.csv', 'pima_2_X_train_res_ds.csv',
              'shuttle-2-vs-5_2_X_train_res_dg.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\ds和dg\\"
save_dir = "./results/"

plot_single_group(
    dataset_names, titles, file_names, root_path, save_dir,
    suffix="mtgp",
    legend_rule=lambda t: t == 'D4',
)
