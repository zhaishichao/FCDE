from common import plot_grid

# ======== 数据配置 ========
titles = [r'w/o $g_{1}$ and $g_{2}$', r'w/o $g_{3}$']
file_names = ['wisconsin_9_X_train_res_border.csv',
              'wisconsin_2_X_train_res_re_g4.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\woC\\"
save_dir = "./results/"

plot_grid(
    titles, file_names, root_path, save_dir,
    output_name="wisconsin_woc_2.pdf",
    ncols=2, figsize=(12, 6), legend_loc="lower right",
)
