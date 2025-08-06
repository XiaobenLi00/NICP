import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import os
import glob
from matplotlib.colors import LinearSegmentedColormap


def main():
    work_dir = "output/matchAMASS_4D-DRESS/4d-dress_gen_eq_pred_inner_points"
    exp_name = work_dir.split("/")[-1]
    morandi_colors = ["#F2D0A9", "#D98E73", "#B24C33"]
    morandi_cmap = mcolors.LinearSegmentedColormap.from_list(
        "morandi", morandi_colors, N=100
    )

    # 获取所有txt文件

    # 定义要查找的文件模式
    file_patterns = [
        "*cd_l2_score*",
        "*v2v_error_raw*",
        "*v2v_error_cham*",
        "*mpjpe_error_raw*",
        "*mpjpe_error_cham*",
    ]

    metrics_names = [
        "CD",
        "V2V",
        "V2V_Cham",
        "MPJPE",
        "MPJPE_Cham",
    ]

    # 存储每个文件的指标列表
    metrics_lists = {}

    # 遍历每个文件模式
    for idx, pattern in enumerate(file_patterns):
        # 查找匹配的文件
        matching_files = glob.glob(os.path.join(work_dir, f"{pattern}.txt"))

        if matching_files:
            file_path = matching_files[0]  # 取第一个匹配的文件
            metrics = []

            # 读取文件前1021行
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 1943:  # 只读取前1021行
                        break

                    # 解析每行：id + 空格 + 指标值
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # 提取指标值（最后一个部分）
                        metric_value = float(parts[-1])
                        metrics.append(metric_value)

            # 存储指标列表，使用文件名作为key
            file_name = os.path.basename(file_path)
            metrics_lists[metrics_names[idx]] = metrics

            print(f"读取文件 {file_name}: {len(metrics)} 个指标")
        else:
            print(f"未找到匹配模式 {pattern} 的文件")
    # for key, value in metrics_lists.items():
    #     print(key, value)
    # 将metrics_lists转换为DataFrame

    # 创建DataFrame
    df = pd.DataFrame(metrics_lists)

    # 计算Spearman相关系数矩阵
    corr_matrix = df.corr(method="spearman")

    # print("Spearman相关系数矩阵:")
    # print(corr_matrix)
    # print(df.columns.tolist())
    features = df.columns.tolist()

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_aspect("equal")
    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=morandi_cmap,
        xticklabels=features,
        yticklabels=features,
        vmin=0.4,
        vmax=1,
        center=0.7,
        ax=ax,
        cbar=False,
        linewidths=0,
        linecolor="none",
        square=True,
        rasterized=True,
        # cbar_kws={'label': ''},
        annot_kws={"size": 18},
    )
    tick_colors = ["#E19645", "#871510"]
    for tick in ax.get_xticklabels():
        metric_name = tick.get_text()
        if metric_name == "CD":
            tick.set_color(tick_colors[0])
        else:
            tick.set_color(tick_colors[1])
    for tick in ax.get_yticklabels():
        metric_name = tick.get_text()
        if metric_name == "CD":
            tick.set_color(tick_colors[0])
        else:
            tick.set_color(tick_colors[1])
    plt.tight_layout()

    plt.savefig(
        os.path.join(work_dir, f"{exp_name}_correlation_metrics.pdf"),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    main()
