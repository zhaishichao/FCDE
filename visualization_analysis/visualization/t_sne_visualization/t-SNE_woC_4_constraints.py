from common import plot_grid

# ======== 数据配置 ========
titles = [r'w/o $g_{1}$', r'w/o $g_{2}$ and $g_{3}$', r'w/o $g_{4}$']
file_names = ['wisconsin_5_X_train_res_dg.csv', 'wisconsin_9_X_train_res_border.csv',
              'wisconsin_2_X_train_res_re_g4.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\woC\\"
save_dir = "./results/"

plot_grid(
    titles, file_names, root_path, save_dir,
    output_name="wisconsin_woc.pdf",
    ncols=3, figsize=(18, 6), legend_loc="upper right",
)
