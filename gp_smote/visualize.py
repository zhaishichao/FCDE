import os

from matplotlib import pyplot as plt


def curve_fitting(list_cv, file_path, filename):
    # 你的收敛曲线数据（这里用示例数据，请替换为你的实际数据）

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(list_cv, 'b-o', linewidth=2, markersize=6,
             markerfacecolor='red', markeredgecolor='red', label='cv')

    # 设置图形属性
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('cv', fontsize=12)
    plt.title('Convergence curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 添加一些辅助线
    plt.axhline(y=min(list_cv), color='r', linestyle='--', alpha=0.5,
                label=f'最优值: {min(list_cv):.3f}')

    full_path = os.path.join(file_path, f"{filename}.png")
    # 保存图形
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    # # 显示图形
    # plt.show()