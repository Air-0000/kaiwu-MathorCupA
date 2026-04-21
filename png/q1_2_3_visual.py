"""
问题2可视化脚本
直接使用给定路线生成三张图
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 设置中文字体
plt.rcParams["font.sans-serif"] = ['PingFang SC', 'Arial Unicode MS']
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ========== 给定数据 ==========
# 最优路线
BEST_ROUTE_1 = [0, 13, 2, 15, 14, 6, 5, 8, 7, 11, 10, 1, 9, 3, 12, 4, 0]
BEST_ROUTE_2 = [0, 2, 13, 6, 5, 8, 7, 11, 10, 1, 9, 3, 12, 4, 15, 14, 0]
BEST_ROUTE_3 = [0, 40, 2, 21, 26, 12, 28, 27, 1, 31, 10, 11, 19, 47, 48, 7, 18, 8, 46, 45, 17, 16, 5, 6, 13, 37, 44, 38, 14, 43, 15, 42, 41, 22, 23, 39, 25, 4, 24, 29, 3, 50, 33, 9, 34, 35, 20, 30, 32, 49, 36, 0]
TRAVEL_TIME_1 = 29
TRAVEL_TIME_2 = 31
TRAVEL_TIME_3 = 61
OBJECTIVE_2 = 84121
OBJECTIVE_3 = 4995871

# 各客户时间窗违反详情（与路线顺序对应）
PER_CUSTOMER = [
    {"node": 0, "arrival": 0, "start": 0, "lb": 0, "ub": 0, "early": 0, "late": 0},
    {"node": 2,  "arrival": 2,  "start": 2,  "lb": 3,  "ub": 12, "early": 1,  "late": 0},
    {"node": 13, "arrival": 5,  "start": 5,  "lb": 9,  "ub": 27, "early": 4,  "late": 0},
    {"node": 6,  "arrival": 8,  "start": 8,  "lb": 7,  "ub": 20, "early": 0,  "late": 0},
    {"node": 5,  "arrival": 11, "start": 11, "lb": 3,  "ub": 14, "early": 0,  "late": 0},
    {"node": 8,  "arrival": 15, "start": 15, "lb": 8,  "ub": 18, "early": 0,  "late": 0},
    {"node": 7,  "arrival": 19, "start": 19, "lb": 9,  "ub": 14, "early": 0,  "late": 5},
    {"node": 11, "arrival": 23, "start": 23, "lb": 4,  "ub": 20, "early": 0,  "late": 3},
    {"node": 10, "arrival": 26, "start": 26, "lb": 13, "ub": 20, "early": 0,  "late": 6},
    {"node": 1,  "arrival": 30, "start": 30, "lb": 17, "ub": 26, "early": 0,  "late": 4},
    {"node": 9,  "arrival": 34, "start": 34, "lb": 7,  "ub": 20, "early": 0,  "late": 14},
    {"node": 3,  "arrival": 38, "start": 38, "lb": 14, "ub": 18, "early": 0,  "late": 20},
    {"node": 12, "arrival": 41, "start": 41, "lb": 5,  "ub": 13, "early": 0,  "late": 28},
    {"node": 4,  "arrival": 45, "start": 45, "lb": 9,  "ub": 25, "early": 0,  "late": 20},
    {"node": 15, "arrival": 51, "start": 51, "lb": 4,  "ub": 20, "early": 0,  "late": 31},
    {"node": 14, "arrival": 55, "start": 55, "lb": 4,  "ub": 18, "early": 0,  "late": 37},
    {"node": 0, "arrival": 0, "start": 0, "lb": 0, "ub": 0, "early": 0, "late": 0},
]

# 10个解的目标值（用于对比图）
ALL_OBJECTIVES = [
    (84121, "解1(最优)"),
    (86251, "解2"),
    (87893, "解3"),
    (88359, "解4"),
    (88389, "解5"),
    (88532, "解6"),
    (90004, "解7"),
    (90392, "解8"),
    (90551, "解9"),
    (90761, "解10"),
]


def plot_best_route(output_path,index):
    """
    图1：最优路线顺序图
    """
    if index == 3:
        OBJECTIVE_opt = OBJECTIVE_3
        BEST_ROUTE = BEST_ROUTE_3
        TRAVEL_TIME = TRAVEL_TIME_3
    else:
        BEST_ROUTE = BEST_ROUTE_1 if index == 1 else BEST_ROUTE_2
        TRAVEL_TIME = TRAVEL_TIME_1 if index == 1 else TRAVEL_TIME_2
        OBJECTIVE_opt = OBJECTIVE_2


    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制折线图
    x = range(len(BEST_ROUTE))
    y = BEST_ROUTE
    ax.plot(x, y, marker='o', linewidth=2, markersize=10, color='#2F7E79')

    # 在点上标注客户编号
    for i, node in enumerate(BEST_ROUTE):
        ax.annotate(str(node), (i, node), textcoords="offset points",
                   xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

    # 添加起点和终点标注
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='配送中心(0)')

    ax.set_xlabel("访问顺序位置", fontsize=12)
    ax.set_ylabel("节点编号", fontsize=12)
    if index == 1:
        ax.set_title(f"第1问最优路线顺序（运输时间={TRAVEL_TIME}分钟）", fontsize=14)
    else:
        ax.set_title(f"第{index}问最优路线顺序（目标值={OBJECTIVE_opt}，运输时间={TRAVEL_TIME}分钟）", fontsize=14)
    if index != 3:
        ax.set_xticks(x)
        ax.set_yticks(range(0, 17))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc='upper right')

    # 标注起点和终点
    ax.text(0, 0, '起点0', ha='left', va='bottom', fontsize=9, color='gray')
    ax.text(len(BEST_ROUTE)-1, 0, '终点0', ha='right', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=280, bbox_inches='tight')
    plt.close()
    print(f"图1已保存: {output_path}")


def plot_seed_compare(output_path):
    """
    图2：各初始解方法目标值对比
    """
    # 按目标值排序
    sorted_data = sorted(ALL_OBJECTIVES, key=lambda x: x[0])
    objectives, labels = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(objectives))
    colors = ['#2F7E79' if i == 0 else '#4C78A8' for i in range(len(objectives))]

    bars = ax.bar(x, objectives, color=colors)

    # 添加数值标签
    for bar, obj in zip(bars, objectives):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 200,
                f'{obj}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("解的编号", fontsize=12)
    ax.set_ylabel("目标值", fontsize=12)
    ax.set_title("第2问各解目标值对比（已按目标值排序）", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)

    # 标注最优解
    ax.axhline(y=objectives[0], color='#E15759', linestyle='--', linewidth=1.5,
               label=f'最优目标值: {objectives[0]}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=280, bbox_inches='tight')
    plt.close()
    print(f"图2已保存: {output_path}")


def plot_violations(output_path):
    """
    图3：各客户时间窗违反情况
    """
    TRAVEL_TIME = TRAVEL_TIME_2
    # 提取数据
    customers = [c["node"] for c in PER_CUSTOMER]
    early = [c["early"] for c in PER_CUSTOMER]
    late = [c["late"] for c in PER_CUSTOMER]

    x = np.arange(len(customers))

    fig, ax = plt.subplots(figsize=(14, 6))

    width = 0.35
    bars1 = ax.bar(x - width/2, early, width, label='早到违反(分钟)', color='#4C78A8')
    bars2 = ax.bar(x + width/2, late, width, label='晚到违反(分钟)', color='#E15759')

    ax.set_xlabel("客户编号", fontsize=12)
    ax.set_ylabel("违反时间(分钟)", fontsize=12)
    ax.set_title(f"第2问各客户时间窗违反情况（最优解，运输时间={TRAVEL_TIME}分钟）", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(customers, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)

    # 添加时间窗信息
    time_windows = [f"[{c['lb']},{c['ub']}]" for c in PER_CUSTOMER]
    for i, tw in enumerate(time_windows):
        ax.text(i, -2, tw, ha='center', va='top', fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=280, bbox_inches='tight')
    plt.close()
    print(f"图3已保存: {output_path}")


def main():

    # 创建输出目录
    output_dir = Path(r"output")
    output_dir.mkdir(exist_ok=True)

    index = input()

    # 生成三张图
    print("\n生成图1: 最优路线顺序图...")
    name = "q" + str(index) + "_best_route.png"
    plot_best_route(output_dir / name,int(index))

    print("\n生成图2: 各解目标值对比...")
    plot_seed_compare(output_dir / "q2_seed_compare.png")

    print("\n生成图3: 时间窗违反情况...")
    plot_violations(output_dir / "q2_violations.png")

    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()